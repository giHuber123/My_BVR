import jsbsim
import numpy as np
import math
import os
import gymnasium as gym
import gc
from gymnasium import spaces
from baseline_action import FlightActions

class AirCombatEnv(gym.Env):
    def __init__(self, model_name="f16", dt=1 / 60.0):
        super(AirCombatEnv, self).__init__()
        self.dt = dt
        self.model_name = model_name
        self.root_path = os.environ.get('JSBSIM_ROOT', 'jsbsim-master')

        # 动作空间：[滚转, 俯仰, 偏航, 油门]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # 观测空间升级为 23 维：
        # 1-15: 本机状态 (高度, 速度, 姿态, 攻角, 过载, 舵面)
        # 16-23: 增强相对态势 (高度差, 速度差, 距离, 相对方向矢量XYZ, ATA正余弦)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)

        self.fdm_red = None
        self.fdm_blue = None
        self.prev_action = np.zeros(4)
        self.prev_prev_action = np.zeros(4)
        self.steps = 0

    def _create_fdm(self):
        fdm = jsbsim.FGFDMExec(self.root_path)
        fdm.set_debug_level(0)
        fdm.load_model(self.model_name)
        fdm.set_dt(self.dt)
        return fdm

    def _init_aircraft(self, fdm, lat_off, lon_off, alt, heading, speed):
        fdm['ic/lat-gc-deg'] = 30.0 + lat_off
        fdm['ic/long-gc-deg'] = 120.0 + lon_off
        fdm['ic/h-sl-ft'] = alt / 0.3048
        fdm['ic/psi-true-deg'] = heading
        fdm['ic/vt-fps'] = speed / 0.3048
        fdm['propulsion/set-running'] = -1
        fdm.run_ic()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.close()

        self.fdm_red = self._create_fdm()
        self.fdm_blue = self._create_fdm()

        # 初始化：红方朝北，蓝方在北边 0.15 度处朝南对冲
        self._init_aircraft(self.fdm_red, 0.0, 0.0, 20000.0, 0.0, 300.0)
        self._init_aircraft(self.fdm_blue, 0.15, 0.0, 20000.0, 180.0, 300.0)

        self.prev_action = np.zeros(4)
        self.prev_prev_action = np.zeros(4)
        self.steps = 0

        return self._get_full_obs(), {}

    def step(self, action_red, action_blue):
        self._apply_action(self.fdm_red, action_red)
        self._apply_action(self.fdm_blue, action_blue)

        self.fdm_red.run()
        self.fdm_blue.run()
        self.steps += 1

        obs = self._get_full_obs()
        reward = self._compute_reward(self.fdm_red, self.fdm_blue, action_red)

        terminated = False
        info = {"reason": "alive"}

        # 终止条件：坠毁或严重失速
        if (self.fdm_red['position/h-sl-ft'] * 0.3048) < 500.0:
            terminated = True
            info["reason"] = "CRASH"
        if abs(self.fdm_red['aero/alpha-deg']) > 25.0:
            terminated = True
            info["reason"] = "STALL"

        return obs, reward, terminated, False, info

    def close(self):
        if self.fdm_red is not None:
            del self.fdm_red
            self.fdm_red = None
        if self.fdm_blue is not None:
            del self.fdm_blue
            self.fdm_blue = None
        gc.collect()

    def _apply_action(self, fdm, action):
        fdm['fcs/aileron-cmd-norm'] = action[0]
        fdm['fcs/elevator-cmd-norm'] = action[1]
        fdm['fcs/rudder-cmd-norm'] = action[2]
        fdm['fcs/throttle-cmd-norm'] = (action[3] + 1.0) * 0.5

    def _get_full_obs(self):
        # 1. 本机状态 (15维)
        def get_my_obs(fdm):
            return np.array([
                fdm['position/h-sl-ft'] * 0.3048 / 10000.0,
                fdm['velocities/vt-fps'] * 0.3048 / 340.0,
                fdm['attitude/phi-rad'] / math.pi,
                fdm['attitude/theta-rad'] / (math.pi / 2),
                math.sin(fdm['attitude/psi-rad']),
                math.cos(fdm['attitude/psi-rad']),
                fdm['aero/alpha-rad'] / (math.pi / 6),
                fdm['aero/beta-rad'] / (math.pi / 18),
                fdm['accelerations/n-pilot-x-norm'] / 2.0,
                fdm['accelerations/n-pilot-y-norm'] / 5.0,
                fdm['accelerations/n-pilot-z-norm'] / 9.0,
                fdm['fcs/left-aileron-pos-norm'],
                fdm['fcs/elevator-pos-norm'],
                fdm['fcs/rudder-pos-norm'],
                fdm['fcs/throttle-pos-norm']
            ], dtype=np.float32)

        # 2. 增强相对态势 (8维)
        def get_rel_obs(f_s, f_t):
            pos_s = self._get_pos_neu(f_s)
            pos_t = self._get_pos_neu(f_t)
            rel_pos = pos_t - pos_s
            dist = np.linalg.norm(rel_pos)
            rel_dir = rel_pos / (dist + 1e-6)  # 包含前后、左右、高低的方向矢量

            d_alt = (f_t['position/h-sl-ft'] - f_s['position/h-sl-ft']) * 0.3048
            d_v = (f_t['velocities/vt-fps'] - f_s['velocities/vt-fps']) * 0.3048

            ata = self._calculate_ata(f_s, f_t)

            return np.array([
                d_alt / 1000.0,  # 高低差
                d_v / 340.0,  # 速度差
                dist / 100000.0,  # 远近
                rel_dir[0],  # 北向(前/后感)
                rel_dir[1],  # 东向(左/右感)
                rel_dir[2],  # 天向(高/低感)
                math.sin(ata),  # 进攻指向感 sin
                math.cos(ata)  # 进攻指向感 cos
            ], dtype=np.float32)

        return np.concatenate([get_my_obs(self.fdm_red), get_rel_obs(self.fdm_red, self.fdm_blue)])

    def _get_pos_neu(self, fdm):
        return np.array([
            fdm['position/lat-gc-deg'] * 111000,
            fdm['position/long-gc-deg'] * 111000,
            fdm['position/h-sl-ft'] * 0.3048
        ])

    def _get_vel_neu(self, fdm):
        return np.array([
            fdm['velocities/v-north-fps'],
            fdm['velocities/v-east-fps'],
            -fdm['velocities/v-down-fps']
        ]) * 0.3048

    def _calculate_ata(self, fdm_s, fdm_t):
        """计算天线指向角 (Antenna Train Angle)"""
        pos_s, pos_t = self._get_pos_neu(fdm_s), self._get_pos_neu(fdm_t)
        rel_pos_vec = (pos_t - pos_s) / (np.linalg.norm(pos_t - pos_s) + 1e-6)

        # 机头指向矢量 (Body X-axis in NEU)
        psi, theta = fdm_s['attitude/psi-rad'], fdm_s['attitude/theta-rad']
        head_dir = np.array([
            math.cos(theta) * math.cos(psi),
            math.cos(theta) * math.sin(psi),
            math.sin(theta)
        ])
        return math.acos(np.clip(np.dot(head_dir, rel_pos_vec), -1.0, 1.0))

    # --- 奖励与惩罚系统 ---

    def _compute_reward(self, fdm_s, fdm_t, action_s):
        return self._compute_situation_reward(fdm_s, fdm_t) + \
            self._compute_alpha_penalty(fdm_s) + \
            self._compute_smoothness_penalty(action_s)

    def _compute_situation_reward(self, fdm_s, fdm_t):
        pos_s, pos_t = self._get_pos_neu(fdm_s), self._get_pos_neu(fdm_t)
        vel_s, vel_t = self._get_vel_neu(fdm_s), self._get_vel_neu(fdm_t)
        v_s, v_t = np.linalg.norm(vel_s), np.linalg.norm(vel_t)

        r_dist = np.exp(-((np.linalg.norm(pos_t - pos_s) - 40000) ** 2) / (2 * 20000 ** 2))
        r_alt = np.tanh((pos_s[2] - pos_t[2]) / 2000.0)
        r_vel = 0.5 * np.tanh((v_s - v_t) / 50.0) + 0.5 * np.exp(-((v_s - 320) ** 2) / 20000)

        energy_s = pos_s[2] + (v_s ** 2) / (2 * 9.81)
        energy_t = pos_t[2] + (v_t ** 2) / (2 * 9.81)
        r_energy = np.tanh((energy_s - energy_t) / 5000.0)

        return r_dist + r_alt + r_vel + r_energy

    def _compute_alpha_penalty(self, fdm_s):
        alpha = abs(fdm_s['aero/alpha-deg'])
        penalty = 0.0
        if alpha > 18.0:
            penalty = 0.5 * (alpha - 18.0) ** 2
            if alpha > 25.0: penalty += 100.0
        return -penalty

    def _compute_smoothness_penalty(self, current_action):
        rate_error = np.sum(np.square(current_action - self.prev_action))
        accel_error = np.sum(np.square(current_action - 2 * self.prev_action + self.prev_prev_action))

        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = current_action.copy()

        return -(0.1 * rate_error + 0.05 * accel_error)


import jsbsim
import numpy as np
import math
import os


# --- 增强版 ACMI 录制类：支持双机 ---
class ACMIDualLogger:
    def __init__(self, filename="air_combat_test.acmi"):
        self.file = open(filename, "w", encoding="utf-8")
        self.file.write("FileType=text/acmi/tacview\n")
        self.file.write("FileVersion=2.1\n")
        self.file.write("0,ReferenceTime=2024-01-01T00:00:00Z\n")

    def log_state(self, sim_time, fdm_red, fdm_blue):
        self.file.write(f"#{sim_time:.2f}\n")

        # 记录红方 (ID=1)
        self._write_aircraft(1, fdm_red, "F-16_RED", "Red")
        # 记录蓝方 (ID=2)
        self._write_aircraft(2, fdm_blue, "F-16_BLUE", "Blue")

    def _write_aircraft(self, obj_id, fdm, name, color):
        lat = fdm['position/lat-geod-deg']
        lon = fdm['position/long-gc-deg']
        alt = fdm['position/h-sl-ft'] * 0.3048
        roll = fdm['attitude/phi-deg']
        pitch = fdm['attitude/theta-deg']
        yaw = fdm['attitude/heading-true-rad']
        yaw = np.rad2deg(yaw)
        # T=Lon|Lat|Alt|Roll|Pitch|Yaw
        line = f"{obj_id},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},Name={name},Color={color},Type=Air+FixedWing\n"
        self.file.write(line)

    def close(self):
        self.file.close()


# --- 修改后的 test_env 函数 ---
def test_env():
    # 1. 初始化环境与录制器
    env = AirCombatEnv(model_name="f16")
    obs, _ = env.reset()

    # 初始化控制器和录制器
    controller = FlightActions()
    controller_blue = FlightActions()
    logger = ACMIDualLogger("air_combat_dogfight.acmi")

    print(f"{'Step':<6} | {'Dist(km)':<10} | {'ATA(deg)':<8} | {'Status':<10}")
    print("-" * 50)

    # 模拟运行 1200 步 (约 20 秒，dt=1/60)
    for i in range(12000):
        # --- 红方逻辑：使用 PID 保持攻角和滚转 ---
        curr_alpha = env.fdm_red['aero/alpha-deg']

        curr_roll = env.fdm_red['attitude/phi-deg']
        curr_q = env.fdm_red['velocities/q-rad_sec']
        curr_p = env.fdm_red['velocities/p-rad_sec']

        curr_alpha_blue = env.fdm_blue['aero/alpha-deg']
        curr_roll_blue = env.fdm_blue['attitude/phi-deg']
        curr_q_blue = env.fdm_blue['velocities/q-rad_sec']
        curr_p_blue = env.fdm_blue['velocities/p-rad_sec']
        # 获取红方 Action (维持 2.5度攻角平飞)
        pid_action = controller.get_action(
            current_alpha=curr_alpha,
            current_roll=curr_roll,
            q_rad=curr_q,
            p_rad=curr_p,
            target_alpha=0.1,
            target_roll=0.0
        )

        pid_action_blue = controller_blue.get_action(
            current_alpha=curr_alpha_blue,
            current_roll=curr_roll_blue,
            q_rad=curr_q_blue,
            p_rad=curr_p_blue,
            target_alpha=0.0,
            target_roll=0.0
        )
        # 组装 Action [aileron, elevator, rudder, throttle]
        # 注意：env.step 期望的是红蓝双方的动作
        action_red = np.array([pid_action[1], -pid_action[0], 0.0, 0.8], dtype=np.float32)

        # 蓝方逻辑：保持现状，油门 0.5
        action_blue = np.array([pid_action_blue[1], -pid_action_blue[0], 0.0, 0.5], dtype=np.float32)

        # 运行环境
        obs, reward, terminated, truncated, info = env.step(action_red, action_blue)

        # --- 录制数据 (每 6 步录一次，即 10Hz) ---
        if i % 6 == 0:
            logger.log_state(env.fdm_red.get_sim_time(), env.fdm_red, env.fdm_blue)

        if terminated:
            print(f"仿真终止: {info['reason']}")
            break

    logger.close()
    env.close()
    print("录制完成：air_combat_dogfight.acmi")


if __name__ == "__main__":
    test_env()