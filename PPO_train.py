import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from env import AirCombatEnv  # 假设你上面的环境存为 air_combat_env.py
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def train():
    def make_env():
        return AirCombatEnv()

    # 1. 创建基础环境
    venv = DummyVecEnv([make_env])

    # 2. 加入自动归一化包装器
    # norm_obs: 自动将观测值归一化（通常到均值0，方差1左右）
    # norm_reward: 自动对奖励进行缩放（对 PPO 的收敛极其重要）
    # clip_obs: 裁剪异常大的观测值，防止你刚才遇到的 NaN 崩溃
    env = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0
    )

    # 2. 定义 PPO 模型
    # 超参数解释：
    # learning_rate: 飞行控制需要精细调节，建议由 3e-4 调小至 1e-4
    # n_steps: 每次更新前收集的步数，1024-2048 比较适合 60Hz 的环境
    # batch_size: 每次梯度下降的样本数
    # ent_coef: 熵系数，增加探索度，防止 AI 很快就只会飞直线
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./ppo_air_combat_tensorboard/"
    )

    # 3. 设置保存点的 Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/",
        name_prefix="ppo_f16_dogfight"
    )

    # 4. 开始训练
    print("开始训练 PPO 代理...")
    model.learn(
        total_timesteps=1000000,  # 建议初次训练 100万步看效果
        callback=checkpoint_callback,
        progress_bar=True
    )

    # 4. 重要：保存模型时必须同时保存归一化器的状态！
    # 否则测试时，AI 会因为无法正确缩放输入而乱飞
    model.save("ppo_f16_dogfight_final")
    env.save("vec_normalize.pkl")


if __name__ == "__main__":
    train()