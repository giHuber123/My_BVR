import numpy as np
import time
import os
import pandas as pd
from env import AirCombatEnv


def test_run_with_debug_export():
    print("正在初始化环境并准备导出物理数据...")

    acmi_path = "./BVR_Test_Flight.acmi"
    excel_path = "./Flight_Data_Comparison.xlsx"

    if os.path.exists(acmi_path): os.remove(acmi_path)

    try:
        env = AirCombatEnv(model_name="f16")
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    obs, _ = env.reset()

    # 建立一个列表来存储每一帧的详细对比数据
    debug_data = []

    try:
        for i in range(10000):  # 运行 1000 步
            # 使用完全相同的随机动作
            action_red = env.action_space.sample()
            action_blue = env.action_space.sample()
            # 获取距离
            dist = obs[17] * 100000.0

            # 距离小于 40km 时，第 200 步发射一颗导弹
            fire = (i == 200 and dist < 40000)

            # 修改后 (适配 Gymnasium 标准)：
            obs, reward, terminated, truncated, info = env.step(action_red, action_blue, fire_red=fire)


            # 逻辑判断也要同步修改：
            done = terminated or truncated
            # --- 核心采集逻辑：直接从环境的 FDM 实例中抓取原始物理量 ---
            def get_fdm_state(fdm):
                return {
                    "Alt_ft": fdm['position/h-sl-ft'],
                    "V_fps": fdm['velocities/vt-fps'],
                    "Alpha_deg": fdm['aero/alpha-deg'],
                    "Beta_deg": fdm['aero/beta-deg'],
                    "Phi_deg": fdm['attitude/phi-deg'],  # 滚转
                    "Theta_deg": fdm['attitude/theta-deg'],  # 俯仰
                    "Psi_deg": fdm['attitude/psi-deg'],  # 航向
                    "Elevator_Pos": fdm['fcs/elevator-pos-norm'],
                    "Aileron_Pos": fdm['fcs/left-aileron-pos-norm'],
                    "NZ": fdm['accelerations/n-pilot-z-norm']
                }

            red_state = get_fdm_state(env.fdm_red)
            blue_state = get_fdm_state(env.fdm_blue)

            # 记录两者的差值
            record = {"step": i}
            for key in red_state.keys():
                record[f"RED_{key}"] = red_state[key]
                record[f"BLUE_{key}"] = blue_state[key]
                record[f"DIFF_{key}"] = red_state[key] - blue_state[key]

            debug_data.append(record)

            if i % 6 == 0:
                env.render(mode="txt", filepath=acmi_path)

            if i % 100 == 0:
                print(f"Progress: {i} steps... Diff Alpha: {record['DIFF_Alpha_deg']:.4f}")

            if done:
                print(info)
                break

    finally:
        env.close()

    # --- 导出到 Excel ---
    df = pd.DataFrame(debug_data)
    df.to_excel(excel_path, index=False)
    print("-" * 30)
    print(f"数据分析完成！")
    print(f"ACMI 录像: {acmi_path}")
    print(f"物理数据对比表: {excel_path}")
    print("请查看 Excel 中的 DIFF_xxx 列，数值不为 0 的项即为导致不一致的源头。")


if __name__ == "__main__":
    test_run_with_debug_export()