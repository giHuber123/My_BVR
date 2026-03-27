import jsbsim
import numpy as np
import math
import os
import gymnasium as gym
import gc
from gymnasium import spaces
from baseline_action import FlightActions

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

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)

        self.max_steps = 12000
        self.gun_fire_timer = 0
        self.kill_threshold = 60  # 1秒连续咬尾判定

        self.fdm_red = None
        self.fdm_blue = None

        self.prev_action = np.zeros(4)
        self.prev_prev_action = np.zeros(4)

        self.ctrl_red = FlightActions()
        self.steps = 0

    def _get_pos_neu(self, fdm):
        """将经纬高转换为本地NEU坐标(米)"""
        lat = fdm['position/lat-gc-deg']
        lon = fdm['position/long-gc-deg']
        alt = fdm['position/h-sl-ft'] * 0.3048
        x = lat * 111132.0
        y = lon * 111132.0 * math.cos(math.radians(lat))
        return np.array([x, y, alt])

    def check_gun_wez(self, fdm_s, fdm_t):
        """检查 WEZ 条件"""
        pos_s = self._get_pos_neu(fdm_s)
        pos_t = self._get_pos_neu(fdm_t)
        rel_pos = pos_t - pos_s
        dist = np.linalg.norm(rel_pos)
        rel_dir = rel_pos / (dist + 1e-6)

        # ATA (指向角)
        psi, theta = fdm_s['attitude/psi-rad'], fdm_s['attitude/theta-rad']
        head_dir = np.array([math.cos(theta) * math.cos(psi), math.cos(theta) * math.sin(psi), math.sin(theta)])
        ata = math.acos(np.clip(np.dot(head_dir, rel_dir), -1.0, 1.0))

        # AA (被咬尾角)
        psi_t, theta_t = fdm_t['attitude/psi-rad'], fdm_t['attitude/theta-rad']
        target_head_dir = np.array(
            [math.cos(theta_t) * math.cos(psi_t), math.cos(theta_t) * math.sin(psi_t), math.sin(theta_t)])
        aa = math.acos(np.clip(np.dot(target_head_dir, rel_dir), -1.0, 1.0))

        in_wez = (300 < dist < 1000) and (math.degrees(ata) < 15.0) and (math.degrees(aa) < 60.0)
        return in_wez, dist, math.degrees(ata)

    def _get_full_obs(self):
        def get_my_obs(fdm):
            return [
                fdm['position/h-sl-ft'] * 0.3048 / 10000.0, fdm['velocities/vt-fps'] * 0.3048 / 340.0,
                fdm['attitude/phi-rad'] / math.pi, fdm['attitude/theta-rad'] / (math.pi / 2),
                math.sin(fdm['attitude/psi-rad']), math.cos(fdm['attitude/psi-rad']),
                fdm['aero/alpha-rad'] / (math.pi / 6), fdm['aero/beta-rad'] / (math.pi / 18),
                fdm['accelerations/n-pilot-x-norm'] / 2.0, fdm['accelerations/n-pilot-y-norm'] / 5.0,
                fdm['accelerations/n-pilot-z-norm'] / 9.0, fdm['fcs/left-aileron-pos-norm'],
                fdm['fcs/elevator-pos-norm'], fdm['fcs/rudder-pos-norm'], fdm['fcs/throttle-pos-norm']
            ]

        pos_s, pos_t = self._get_pos_neu(self.fdm_red), self._get_pos_neu(self.fdm_blue)
        rel_pos = pos_t - pos_s
        dist = np.linalg.norm(rel_pos)
        rel_dir = rel_pos / (dist + 1e-6)

        d_alt = pos_t[2] - pos_s[2]
        d_v = (self.fdm_blue['velocities/vt-fps'] - self.fdm_red['velocities/vt-fps']) * 0.3048

        # 计算辅助 ATA 用语观测
        psi, theta = self.fdm_red['attitude/psi-rad'], self.fdm_red['attitude/theta-rad']
        head_dir = np.array([math.cos(theta) * math.cos(psi), math.cos(theta) * math.sin(psi), math.sin(theta)])
        ata = math.acos(np.clip(np.dot(head_dir, rel_dir), -1.0, 1.0))

        rel_obs = [d_alt / 1000.0, d_v / 340.0, dist / 10000.0, rel_dir[0], rel_dir[1], rel_dir[2], math.sin(ata),
                   math.cos(ata)]
        return np.concatenate([get_my_obs(self.fdm_red), rel_obs]).astype(np.float32)

    def step(self, action_blue):

        act_r = self.ctrl_red.get_action(self.fdm_red['aero/alpha-deg'],
        self.fdm_red['attitude/phi-deg'],self.fdm_red['velocities/q-rad_sec'], self.fdm_red['velocities/p-rad_sec'], 1, 0.0)
        action_red = np.array([act_r[1], act_r[0], 0.0, 0.8], dtype=np.float32)
        self._apply_action(self.fdm_blue, action_blue)
        self._apply_action(self.fdm_red, action_red)
        self.fdm_red.run()
        self.fdm_blue.run()
        self.steps += 1

        in_wez, dist, ata_deg = self.check_gun_wez(self.fdm_red, self.fdm_blue)
        if in_wez:
            self.gun_fire_timer += 1
        else:
            self.gun_fire_timer = max(0, self.gun_fire_timer - 1)

        obs = self._get_full_obs()
        reward = self._compute_reward(self.fdm_red, self.fdm_blue, action_red)

        terminated = False
        truncated = False
        info = {"reason": "alive", "dist": dist, "ata": ata_deg}

        if self.gun_fire_timer >= self.kill_threshold:
            terminated, info["reason"] = True, "KILLED_TARGET"
            reward += 1000.0
        elif (self.fdm_red['position/h-sl-ft'] * 0.3048) < 500.0:
            terminated, info["reason"] = True, "CRASH"
        elif abs(self.fdm_red['aero/alpha-deg']) > 25.0:
            terminated, info["reason"] = True, "STALL"
        elif self.steps >= self.max_steps:
            truncated, info["reason"] = True, "TIMEOUT"

        return obs, reward, terminated, truncated, info

    def _apply_action(self, fdm, action):
        fdm['fcs/aileron-cmd-norm'] = action[0]
        fdm['fcs/elevator-cmd-norm'] = action[1]
        fdm['fcs/rudder-cmd-norm'] = action[2]
        fdm['fcs/throttle-cmd-norm'] = (action[3] + 1.0) * 0.5

    # def check_gun_wez(self, fdm_s, fdm_t):
    #     """
    #     检查进攻方(fdm_s)是否在防守方(fdm_t)的机炮攻击区内
    #     """
    #     pos_s = np.array([fdm_s['position/lat-gc-deg'] * 111000, fdm_s['position/long-gc-deg'] * 111000,
    #                       fdm_s['position/h-sl-ft'] * 0.3048])
    #     pos_t = np.array([fdm_t['position/lat-gc-deg'] * 111000, fdm_t['position/long-gc-deg'] * 111000,
    #                       fdm_t['position/h-sl-ft'] * 0.3048])
    #
    #     # 1. 计算距离 (单位: 米)
    #     dist = np.linalg.norm(pos_t - pos_s)
    #
    #     # 2. 计算 ATA (机头指向敌机的角度)
    #     rel_pos_vec = (pos_t - pos_s) / (dist + 1e-6)
    #     psi, theta = fdm_s['attitude/psi-rad'], fdm_s['attitude/theta-rad']
    #     head_dir = np.array([math.cos(theta) * math.cos(psi), math.cos(theta) * math.sin(psi), math.sin(theta)])
    #     ata = math.acos(np.clip(np.dot(head_dir, rel_pos_vec), -1.0, 1.0))
    #
    #     # 3. 计算 Aspect Angle (敌机尾后相对于你的角度)
    #     psi_t, theta_t = fdm_t['attitude/psi-rad'], fdm_t['attitude/theta-rad']
    #     tail_dir = np.array(
    #         [math.cos(theta_t) * math.cos(psi_t), math.cos(theta_t) * math.sin(psi_t), math.sin(theta_t)])
    #     # 注意：AA 是敌机速度矢量与相对位置矢量的夹角
    #     aa = math.acos(np.clip(np.dot(tail_dir, rel_pos_vec), -1.0, 1.0))
    #
    #     # 逻辑判断
    #     in_range = 300 < dist < 1000
    #     in_ata = math.degrees(ata) < 15.0  # 机炮瞄准要求极高
    #     in_aa = math.degrees(aa) < 45.0  # 处于敌机尾后区
    #
    #     return in_range and in_ata and in_aa, dist, math.degrees(ata)

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

    def _compute_gun_wez_reward(self, fdm_s, fdm_t):
        """
        计算机炮攻击区(WEZ)的塑造奖励
        fdm_s: 进攻方 (Red)
        fdm_t: 防守方 (Blue)
        """
        # 1. 提取基础物理量
        pos_s = self._get_pos_neu(fdm_s)
        pos_t = self._get_pos_neu(fdm_t)
        rel_pos = pos_t - pos_s
        dist = np.linalg.norm(rel_pos)
        rel_dir = rel_pos / (dist + 1e-6)

        # 2. 计算关键角度 (ATA 和 AA)
        # ATA: 机头指向敌机的角度 (越小越好)
        ata = self._calculate_ata(fdm_s, fdm_t)

        # AA (Aspect Angle): 敌机尾部指向你的角度 (越小代表你越在它正后方)
        psi_t, theta_t = fdm_t['attitude/psi-rad'], fdm_t['attitude/theta-rad']
        # 敌机机头方向向量
        target_head_dir = np.array([
            math.cos(theta_t) * math.cos(psi_t),
            math.cos(theta_t) * math.sin(psi_t),
            math.sin(theta_t)
        ])
        # AA 是敌机速度矢量与“敌机到我”矢量的夹角
        # 这里我们简化为：你的位置与敌机机头方向的夹角。
        # 如果 dot(target_head_dir, -rel_dir) 接近 1，说明你在它正后方
        aspect_cos = np.dot(target_head_dir, rel_dir)
        aa = math.acos(np.clip(aspect_cos, -1.0, 1.0))

        # --- 开始塑造奖励 ---

        # A. 指向奖励 (Angle Reward): 鼓励机头对准目标
        # 使用 cos(ata)，当 ata=0时奖励为1，ata=90时为0，ata>90为负
        r_ata = math.cos(ata)

        # B. 距离奖励 (Distance Reward): 鼓励保持在 300-1000米
        # 使用高斯分布，峰值在 600米，标准差 400米
        r_dist = math.exp(-((dist - 600) ** 2) / (2 * 400 ** 2))

        # C. 咬尾奖励 (Aspect Reward): 鼓励绕到对方后方
        # 当你在对方后方时 (aspect_cos > 0)，奖励为正
        r_aspect = aspect_cos

        # D. 综合射击窗口奖励 (Gun WEZ Bonus)
        # 只有当同时满足：距离近、角度准、在后方时，给一个巨大的乘积奖励
        is_in_wez = (300 < dist < 1000) and (math.degrees(ata) < 15) and (math.degrees(aa) < 60)
        r_wez_bonus = 10.0 if is_in_wez else 0.0

        # 加权求和 (权重可以根据训练效果调整)
        total_gun_reward = (0.5 * r_ata) + (0.3 * r_dist) + (0.2 * r_aspect) + r_wez_bonus

        return total_gun_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.fdm_red: self.close()
        self.fdm_red, self.fdm_blue = self._create_fdm(), self._create_fdm()
        self._init_aircraft(self.fdm_red, 0.0, 0.0, 20000.0, 0.0, 300.0)
        self._init_aircraft(self.fdm_blue, 0.15, 0.0, 20000.0, 180.0, 300.0)
        self.steps, self.gun_fire_timer = 0, 0
        self.prev_action = np.zeros(4)
        self.prev_prev_action = np.zeros(4)
        self.ctrl_red.reset()

        return self._get_full_obs(), {}

    def _create_fdm(self):
        fdm = jsbsim.FGFDMExec(self.root_path)
        fdm.set_debug_level(0)
        fdm.load_model(self.model_name)
        fdm.set_dt(self.dt)
        return fdm

    def _init_aircraft(self, fdm, lat_off, lon_off, alt, heading, speed):
        fdm['ic/lat-gc-deg'], fdm['ic/long-gc-deg'] = 30.0 + lat_off, 120.0 + lon_off
        fdm['ic/h-sl-ft'], fdm['ic/psi-true-deg'] = alt / 0.3048, heading
        fdm['ic/vt-fps'] = speed / 0.3048
        fdm['propulsion/set-running'] = -1
        fdm.run_ic()

    def close(self):
        if self.fdm_red: del self.fdm_red
        if self.fdm_blue: del self.fdm_blue
        self.fdm_red, self.fdm_blue = None, None
        gc.collect()


# --- 录制与测试保持你原来的 ACMIDualLogger 和 test_env 逻辑即可 ---
# --- 录制类 ---
class ACMIDualLogger:
    def __init__(self, filename="air_combat_dogfight.acmi"):
        self.file = open(filename, "w", encoding="utf-8")
        self.file.write("FileType=text/acmi/tacview\nFileVersion=2.1\n0,ReferenceTime=2024-01-01T00:00:00Z\n")

    def log_state(self, sim_time, fdm_red, fdm_blue):
        self.file.write(f"#{sim_time:.2f}\n")
        self._write_aircraft(1, fdm_red, "F-16_RED", "Red")
        self._write_aircraft(2, fdm_blue, "F-16_BLUE", "Blue")

    def _write_aircraft(self, obj_id, fdm, name, color):
        line = f"{obj_id},T={fdm['position/long-gc-deg']}|{fdm['position/lat-geod-deg']}|{fdm['position/h-sl-ft'] * 0.3048}|{fdm['attitude/phi-deg']}|{fdm['attitude/theta-deg']}|{np.rad2deg(fdm['attitude/psi-rad'])},Name={name},Color={color},Type=Air+FixedWing\n"
        self.file.write(line)

    def close(self):
        self.file.close()


# --- 测试主函数 ---
def test_env():
    env = AirCombatEnv()
    obs, _ = env.reset()
    logger = ACMIDualLogger()
    ctrl_red, ctrl_blue = FlightActions(), FlightActions()

    for i in range(12000):
        # 蓝方控制 (平飞)
        act_b = ctrl_blue.get_action(env.fdm_blue['aero/alpha-deg'], env.fdm_blue['attitude/phi-deg'],
                                     env.fdm_blue['velocities/q-rad_sec'], env.fdm_blue['velocities/p-rad_sec'], 0.0,
                                     0.0)

        # 映射到环境：[aileron, elevator, rudder, throttle]

        step_act_b = np.array([act_b[1], -act_b[0], 0.0, 0.5], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(step_act_b)

        if i % 6 == 0: logger.log_state(env.fdm_red.get_sim_time(), env.fdm_red, env.fdm_blue)
        if i % 60 == 0:
            print(f"Step: {i} | Dist: {info['dist']:.1f}m | Timer: {env.gun_fire_timer} | Status: {info['reason']}")

        if terminated or truncated:
            print(f"Simulation Ended: {info['reason']}")
            break

    logger.close()
    env.close()


if __name__ == "__main__":
    test_env()