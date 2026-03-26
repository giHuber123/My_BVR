import jsbsim
import numpy as np
import math
import os
import gymnasium as gym
from gymnasium import spaces

class UnitreeStyleReward:
    """宇树风格平滑奖励"""

    def __init__(self, action_dim, sigma=0.1, weight=0.2):
        self.prev_actions = np.zeros(action_dim)
        self.sigma = sigma
        self.weight = weight

    def compute_action_smooth_reward(self, current_actions):
        action_error = np.sum(np.square(current_actions - self.prev_actions))
        smooth_reward = self.weight * np.exp(-action_error / self.sigma)
        self.prev_actions = current_actions.copy()
        return smooth_reward


class AirCombatEnv(gym.Env):
    def __init__(self, model_name="f16", dt=1 / 60.0):
        super(AirCombatEnv, self).__init__()
        self.dt = dt
        self.model_name = model_name
        self.root_path = os.environ.get('JSBSIM_ROOT', 'jsbsim-master')

        # 动作空间：[滚转, 俯仰, 偏航, 油门]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # 观测空间：19维
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)

        self.smooth_handler = UnitreeStyleReward(action_dim=4)
        self.steps = 0
        self.fdm_red = None
        self.fdm_blue = None
        self.missiles = []
        self._create_records = False

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
        super().reset(seed=seed)  # 适配 gymnasium 标准
        self.close()
        self.fdm_red = self._create_fdm()
        self.fdm_blue = self._create_fdm()

        # 初始化位置 (恢复对冲布局或你需要的测试布局)
        self._init_aircraft(self.fdm_red, 0.0, 0.0, 6000.0, 0.0, 250.0)
        self._init_aircraft(self.fdm_blue, 0.15, 0.0, 6000.0, 180.0, 250.0)

        # 关键：状态对齐
        trim_props = [
            'fcs/elevator-pos-norm', 'fcs/left-aileron-pos-norm',
            'fcs/right-aileron-pos-norm', 'fcs/rudder-pos-norm',
            'propulsion/engine/thrust-lbs', 'propulsion/engine/n2',
            'aero/alpha-rad', 'aero/beta-rad'
        ]
        for prop in trim_props:
            self.fdm_blue[prop] = self.fdm_red[prop]

        self.steps = 0
        self.missiles = []
        self._create_records = False
        return self._get_full_obs(), {}  # 返回 obs 和 empty info

    def fire_missile(self, side="RED"):
        """发射导弹逻辑"""
        if side == "RED":
            m = SimpleMissile(self.fdm_red, self.fdm_blue, id=f"RM{len(self.missiles)}")
        else:
            m = SimpleMissile(self.fdm_blue, self.fdm_red, id=f"BM{len(self.missiles)}")
        self.missiles.append(m)

    def step(self, action_red, action_blue, fire_red=False, fire_blue=False):
        """执行物理步进，增加了 fire 参数"""
        # 1. 应用控制指令
        self._apply_action(self.fdm_red, action_red)
        self._apply_action(self.fdm_blue, action_blue)

        # 2. 物理模拟
        self.fdm_red.run()
        self.fdm_blue.run()

        # 3. 处理导弹发射 (修复了之前 NameError 的地方)
        if fire_red:
            self.fire_missile("RED")
        if fire_blue:
            self.fire_missile("BLUE")

        # 4. 更新所有导弹状态
        done_by_missile = False
        missile_reason = "none"
        for m in self.missiles:
            if m.active:
                status = m.update(self.dt)
                if status == "HIT":
                    done_by_missile = True
                    missile_reason = f"MISSILE_HIT_{m.id}"

        self.steps += 1

        # 5. 计算观测和奖励
        obs = self._get_full_obs()
        reward = self._compute_reward(self.fdm_red, self.fdm_blue, action_red)

        # 6. 终止判定
        terminated = False
        truncated = False
        info = {"reason": "none"}

        alpha_deg = abs(obs[6] * 30.0)
        if alpha_deg > 25.0:
            terminated = True
            info["reason"] = "STALL"
        elif obs[0] * 10000.0 < 500.0:
            terminated = True
            info["reason"] = "CRASH"
        elif done_by_missile:
            terminated = True
            info["reason"] = missile_reason
        elif self.steps > 100000:
            truncated = True
            info["reason"] = "TIMEOUT"

        return obs, reward, terminated, truncated, info

    def _apply_action(self, fdm, action):
        fdm['fcs/aileron-cmd-norm'] = action[0]
        fdm['fcs/elevator-cmd-norm'] = action[1]
        fdm['fcs/rudder-cmd-norm'] = action[2]
        fdm['fcs/throttle-cmd-norm'] = action[3]

    def _get_my_obs(self, fdm):
        """本机状态: 15维"""
        ft2m, fps2ms = 0.3048, 0.3048
        alt = fdm['position/h-sl-ft'] * ft2m
        v_true = fdm['velocities/vt-fps'] * fps2ms
        roll, pitch, yaw = fdm['attitude/phi-rad'], fdm['attitude/theta-rad'], fdm['attitude/psi-rad']
        alpha, beta = fdm['aero/alpha-rad'], fdm['aero/beta-rad']
        nx = fdm['accelerations/n-pilot-x-norm']
        ny = fdm['accelerations/n-pilot-y-norm']
        nz = fdm['accelerations/n-pilot-z-norm']

        return np.array([
            alt / 10000.0, v_true / 340.0,
            roll / math.pi, pitch / (math.pi / 2), math.sin(yaw), math.cos(yaw),
            alpha / (math.pi / 6), beta / (math.pi / 18),
            nx / 2.0, ny / 5.0, nz / 9.0,
            fdm['fcs/left-aileron-pos-norm'], fdm['fcs/elevator-pos-norm'],
            fdm['fcs/rudder-pos-norm'], fdm['fcs/throttle-pos-norm']
        ], dtype=np.float32)

    def _get_rel_obs(self, fdm_self, fdm_target):
        """相对状态: 4维 (高度差, 速度差, 距离, 航向差)"""
        # 简化版 NEU 坐标转换逻辑
        pos_self = self._get_pos_neu(fdm_self)
        pos_target = self._get_pos_neu(fdm_target)
        rel_pos = pos_target - pos_self
        dist = np.linalg.norm(rel_pos)

        delta_alt = (fdm_target['position/h-sl-ft'] - fdm_self['position/h-sl-ft']) * 0.3048
        delta_v = (fdm_target['velocities/vt-fps'] - fdm_self['velocities/vt-fps']) * 0.3048

        # 航向差 (带 Side Flag)
        vel_self = self._get_vel_neu(fdm_self)[:2]  # XY面速度
        rel_pos_2d = rel_pos[:2]
        angle_off = math.acos(
            np.clip(np.dot(vel_self, rel_pos_2d) / (np.linalg.norm(vel_self) * np.linalg.norm(rel_pos_2d) + 1e-8), -1,
                    1))
        side_flag = np.sign(np.cross(vel_self, rel_pos_2d))
        delta_heading = angle_off * (side_flag if side_flag != 0 else 1)

        return np.array([delta_alt / 1000.0, delta_v / 340.0, dist / 100000.0, delta_heading / math.pi],
                        dtype=np.float32)

    def _get_full_obs(self):
        """组合 15维本机 + 4维相对状态"""
        # ... (这里保留之前写好的 _get_my_obs 和 _get_rel_obs 逻辑，代码略)
        # 注意：过载项要使用你文档里提到的 n-pilot-z-norm
        my_obs = self._get_my_obs(self.fdm_red)
        rel_obs = self._get_rel_obs(self.fdm_red, self.fdm_blue)
        return np.concatenate([my_obs, rel_obs])

    def render(self, mode="txt", filepath='./BVR_Record.acmi'):
        """手动拼接 ACMI 字符串，确保导弹使用纯数字 ID"""
        if mode == "txt":
            if not self._create_records:
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\nFileVersion=2.1\n0,ReferenceTime=2024-01-01T00:00:00Z\n")
                self._create_records = True

            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                f.write(f"#{self.steps * self.dt:.2f}\n")

                # 1. 写入两机坐标姿态 (101, 102)
                f.write(self._tacview_line("101", self.fdm_red, "RED") + "\n")
                f.write(self._tacview_line("102", self.fdm_blue, "BLUE") + "\n")

                # 2. 导弹显示：使用 enumerate 生成纯数字序列
                for idx, m in enumerate(self.missiles):
                    if m.active:
                        # 分配一个唯一的纯数字 ID，从 201 开始
                        # 201, 202, 203... 这样绝不会和 101, 102 冲突
                        numeric_id = 201 + idx

                        # 格式说明：ID(纯数字), T=经度|纬度|高度, Name, Color, Type
                        # 注意：只用一个逗号连接坐标块和属性块
                        mlf = f"{numeric_id},T={m.pos[1]:.6f}|{m.pos[0]:.6f}|{m.pos[2]:.2f},"

                        # 这里的 Name 可以随意写，但 ID 必须是数字
                        # 增加 Color 区分，如果是红方导弹(假设前几个是红方的)可以用逻辑判断
                        color = "Red" if "RM" in str(m.id) else "Blue"
                        mlf += f"Name=AIM-120C,Color={color},Type=Weapon+Missile"

                        f.write(mlf + "\n")

    def _tacview_line(self, id, fdm, name):
        # 提取数据
        lon = fdm['position/long-gc-deg']
        lat = fdm['position/lat-gc-deg']
        alt = fdm['position/h-sl-ft'] * 0.3048
        phi = fdm['attitude/phi-deg']
        theta = fdm['attitude/theta-deg']
        psi = fdm['attitude/psi-deg']

        # 拼接字符串
        # 注意：Type=Air+FixedWing 告诉 Tacview 这是一个飞机
        # 如果你想区分红蓝，Color 在这里也很有用
        color = "Red" if name == "RED" else "Blue"

        line = f"{id},T={lon:.6f}|{lat:.6f}|{alt:.2f}|{phi:.2f}|{theta:.2f}|{psi:.2f},"
        line += f"Name={name},Color={color},Type=Air+FixedWing+Fighter"
        return line

    # --- 辅助工具函数 ---
    def _apply_action(self, fdm, action):
        fdm['fcs/aileron-cmd-norm'] = action[0]
        fdm['fcs/elevator-cmd-norm'] = action[1]
        fdm['fcs/rudder-cmd-norm'] = action[2]
        fdm['fcs/throttle-cmd-norm'] = action[3]

    def _get_pos_neu(self, fdm):
        # 简化的经纬度转平直坐标 (仅作相对计算)
        return np.array([fdm['position/lat-gc-deg'] * 111000, fdm['position/long-gc-deg'] * 111000,
                         fdm['position/h-sl-ft'] * 0.3048])

    def _get_vel_neu(self, fdm):
        return np.array(
            [fdm['velocities/v-north-fps'], fdm['velocities/v-east-fps'], -fdm['velocities/v-down-fps']]) * 0.3048

    def _calculate_ata(self, fdm_s, fdm_t):
        # 计算 Antenna Train Angle (机头指向与目标夹角)
        v_body = np.array([1, 0, 0])  # 简化假设机体坐标系
        return abs(
            fdm_s['attitude/psi-rad'] - math.atan2(fdm_t['position/long-gc-deg'] - fdm_s['position/long-gc-deg'],
                                                   fdm_t['position/lat-gc-deg'] - fdm_s['position/lat-gc-deg']))

    def close(self):
        self.fdm_red = None
        self.fdm_blue = None

    def _compute_reward(self, fdm_s, fdm_t, action_s):
        obs = self._get_my_obs(fdm_s)
        rel_obs = self._get_rel_obs(fdm_s, fdm_t)
        dist = rel_obs[2] * 100000.0
        delta_heading = rel_obs[3] * math.pi

        # 1. 宇树风格：动作平滑奖励
        r_smooth = self.smooth_handler.compute_action_smooth_reward(action_s)

        # 2. 宇树风格：失速惩罚 (Alpha Constraint)
        # 从 obs[6] 还原 alpha 度数: obs[6] * 30
        alpha_deg = abs(obs[6] * 30.0)
        p_stall = 0.0
        if alpha_deg > 18.0:
            p_stall = 0.5 * (alpha_deg - 18.0) ** 2
        if alpha_deg > 25.0:
            p_stall = 500.0  # 触发硬终止的惩罚量

        # 3. BVR 战术奖励 (Crank Logic)
        ata = self._calculate_ata(fdm_s, fdm_t)
        r_bvr = 0.0
        if dist > 60000:
            r_bvr = 1.0 * np.exp(-abs(delta_heading))  # 远距离追踪
        else:
            # 近距离 Crank 奖励: 鼓励保持在 45度(0.78rad) 左右
            error_crank = abs(delta_heading) - 0.78
            r_bvr = 1.5 * np.exp(-(error_crank ** 2) / 0.1)

        return r_smooth + r_bvr - p_stall


class SimpleMissile:
    def __init__(self, launcher_fdm, target_fdm, id="M1"):
        self.active = True
        self.id = id
        self.target_fdm = target_fdm

        # 初始位置：从发射机位置启动
        self.pos = np.array([
            launcher_fdm['position/lat-gc-deg'],
            launcher_fdm['position/long-gc-deg'],
            launcher_fdm['position/h-sl-ft'] * 0.3048
        ])

        # 初始速度：发射机速度 + 导弹初速 (假设 300m/s)
        self.speed = launcher_fdm['velocities/vt-fps'] * 0.3048 + 300.0
        self.heading = launcher_fdm['attitude/psi-rad']
        self.pitch = launcher_fdm['attitude/theta-rad']

        self.timer = 0
        self.max_time = 40.0  # 动力射程时间

    def update(self, dt):
        if not self.active: return

        # 1. 获取目标位置
        target_pos = np.array([
            self.target_fdm['position/lat-gc-deg'],
            self.target_fdm['position/long-gc-deg'],
            self.target_fdm['position/h-sl-ft'] * 0.3048
        ])

        # 2. 简单的比例导引逻辑 (朝向目标点移动)
        # 这里用简化的经纬度差转向量
        dir_vec = target_pos - self.pos
        # 纬度1度约111km，经度1度在30度纬度约96km
        dist_vec = dir_vec * np.array([111000, 96000, 1])
        dist = np.linalg.norm(dist_vec)

        if dist < 500:  # 命中判定
            self.active = False
            return "HIT"

        # 更新位置 (简化：匀速直线追逐)
        move_dir = dist_vec / (dist + 1e-6)
        self.pos += (move_dir * self.speed * dt) / np.array([111000, 96000, 1])

        self.timer += dt
        if self.timer > self.max_time:
            self.active = False
            return "MISS"
        return "FLYING"
