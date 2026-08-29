import gymnasium as gym
import numpy as np

from amazedex_rps_env import AmazeDexRockPaperScissorsEnv

gym.register(
    id="AmazeDex/RockPaperScissors-v0",
    entry_point=AmazeDexRockPaperScissorsEnv,
    max_episode_steps=200,  # matches MAX_STEPS in amazedex_rps_env.py
)


if __name__ == "__main__":
    env = gym.make("AmazeDex/RockPaperScissors-v0", render_mode="human")

    for episode in range(100):
        obs, info = env.reset(seed=episode)
        total_reward = 0.0
        terminated = truncated = False
        steps = 0

        low = env.action_space.low
        high = env.action_space.high

        while not (terminated or truncated):
            action = low + (np.sin(steps / 20.0) * 0.5 + 0.5) * (high - low)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        print(
            f"episode {episode}: return={total_reward:.2f}, steps={steps}, "
            f"target={info['target_gesture_name']}, "
            f"predicted={info['predicted_gesture_name']}, "
            f"correct={info['correct_gesture']}"
        )

    env.close()
