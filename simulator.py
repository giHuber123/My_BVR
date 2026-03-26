import numpy as np
import math
from pathlib import Path
from datetime import datetime


class MyStandaloneMissile:
    def __init__(self):
        # 物理参数
        self.m0, self.m = 84.0, 84.0
        self.dm, self.Isp, self.t_thrust = 6.0, 120.0, 3.0
        self.nyz_max, self.Rc, self.v_min = 30.0, 300.0, 150.0
        self.cD, self.diameter = 0.4, 0.127
        self.area = math.pi * (self.diameter / 2) ** 2
        self.g, self.K = 9.81, 4.0  # 稍微调高导引系数

        # 初始状态 (单位: 米)
        self.pos = np.array([0.0, 0.0, 6000.0])  # 北, 东, 高
        self.vel = np.array([350.0, 0.0, 0.0])  # 初始速度 350m/s
        self.t = 0.0
        self.status = "FLYING"

    def get_rho(self, alt):
        return 1.225 * math.exp(-max(0, alt) / 9300.0)

    def step(self, target_pos, target_vel, dt):
        if self.status != "FLYING": return

        rel_pos = target_pos - self.pos
        dist = np.linalg.norm(rel_pos)
        rel_vel = target_vel - self.vel
        v_norm = np.linalg.norm(self.vel)

        if dist < self.Rc:
            self.status = "HIT"
            return

        # 比例导引 (PN)
        omega = np.cross(rel_pos, rel_vel) / (dist ** 2 + 1e-6)
        acc_cmd = self.K * np.cross(self.vel, omega)

        # 过载限制
        max_a = self.nyz_max * self.g
        acc_mag = np.linalg.norm(acc_cmd)
        if acc_mag > max_a:
            acc_cmd = (acc_cmd / acc_mag) * max_a

        # 动力学
        thrust_mag = (self.g * self.Isp * self.dm) if self.t < self.t_thrust else 0.0
        drag_mag = 0.5 * self.get_rho(self.pos[2]) * (v_norm ** 2) * self.area * self.cD

        unit_vel = self.vel / (v_norm + 1e-6)
        acc_physics = ((thrust_mag - drag_mag) * unit_vel) / self.m + np.array([0, 0, -self.g])

        self.vel += (acc_physics + acc_cmd) * dt
        self.pos += self.vel * dt
        self.t += dt
        if self.t < self.t_thrust:
            self.m -= self.dm * dt

        if v_norm < self.v_min or self.t > 100.0 or self.pos[2] < 0:
            self.status = "MISS"


def run_test():
    msl = MyStandaloneMissile()
    # 目标：在前方 8km，向西横切
    t_pos = np.array([8000.0, 2000.0, 6000.0])
    t_vel = np.array([0.0, -250.0, 0.0])

    dt = 1 / 30.0  # 提高步长感官
    ref_lat, ref_lon = 30.0, 120.0

    with open("missile_test.txt.acmi", "w", encoding='utf-8') as f:
        # 1. 严格的文件头 (注意 ReferenceTime 的格式)
        f.write("FileType=text/acmi/tacview\nFileVersion=2.1\n")
        f.write(f"0,ReferenceTime=2026-03-26T12:00:00Z,ReferenceLocation={ref_lat},{ref_lon},0\n")

        curr_time = 0.0
        while msl.status == "FLYING":
            # 2. 时间戳必须以 # 开头，独立一行
            f.write(f"#{curr_time:.2f}\n")

            # 经纬度转换
            m_lat = ref_lat + (msl.pos[0] / 111319.0)
            m_lon = ref_lon + (msl.pos[1] / (111319.0 * math.cos(math.radians(ref_lat))))
            t_lat = ref_lat + (t_pos[0] / 111319.0)
            t_lon = ref_lon + (t_pos[1] / (111319.0 * math.cos(math.radians(ref_lat))))

            # 3. 实体更新 (ID 必须一致)
            # 导弹: ID=101, 目标: ID=201
            f.write(f"101,T={m_lon}|{m_lat}|{msl.pos[2]},Name=AIM9,Type=Weapon+Missile,Color=Red\n")
            f.write(f"201,T={t_lon}|{t_lat}|{t_pos[2]},Name=Target,Type=Air+FixedWing,Color=Blue\n")

            msl.step(t_pos, t_vel, dt)
            t_pos += t_vel * dt
            curr_time += dt
            if curr_time > 100: break

        # 爆炸判定
        if msl.status == "HIT":
            f.write(f"#{curr_time:.2f}\n")
            f.write(f"101,Status=Dead\n")
            f.write(f"E01,T={m_lon}|{m_lat}|{msl.pos[2]},Type=Misc+Explosion,Radius=300\n")

    print(f"仿真结束！状态: {msl.status}, 累计时长: {curr_time:.2f}s")


if __name__ == "__main__":
    run_test()