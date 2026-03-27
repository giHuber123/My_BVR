import jsbsim
import numpy as np
import math
import os

import numpy as np
class FlightActions:
    """
    负责计算并返回控制指令 (Actions)
    """

    def __init__(self):
        # 纵向 (Alpha) 参数
        self.alpha_kp, self.alpha_ki, self.alpha_kd = 0.04, 0.01, 0.02
        self.alpha_integral = 0

        # 横向 (Roll) 参数
        self.roll_kp, self.roll_kd = 0.015, 0.01

        self.dt = 1/60 # 80Hz

    def get_action(self, current_alpha, current_roll, q_rad, p_rad, target_alpha, target_roll):
        """
        输入当前传感器数值，输出 [elevator_cmd, aileron_cmd]
        """
        # --- 1. 计算升降舵 Action (纵向) ---
        alpha_error = target_alpha - current_alpha
        self.alpha_integral += alpha_error * self.dt

        # 升降舵输出计算
        elevator_out = (alpha_error * self.alpha_kp) + (self.alpha_integral * self.alpha_ki) - (q_rad * self.alpha_kd)
        # 限制并反转极性 (JSBSim 习惯)
        elevator_action = max(min(-elevator_out, 1.0), -1.0)

        # --- 2. 计算副翼 Action (横向) ---
        roll_error = target_roll - current_roll

        # 副翼输出计算
        aileron_out = (roll_error * self.roll_kp) - (p_rad * self.roll_kd)
        aileron_action = max(min(aileron_out, 1.0), -1.0)

        return [elevator_action, aileron_action]
