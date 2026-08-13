import gymnasium as gym
import numpy as np

from amzedex_cube_env2 import AmazeDexCubeGraspEnv, MAX_STEPS


gym.register(
    id="AmazeDex/CubeGrasp-v0",
    entry_point=AmazeDexCubeGraspEnv,
    # AmazeDexCubeGraspEnv already truncates internally at MAX_STEPS (800) via
    # its own `self.steps >= self.max_steps` check, so we don't want gym's
    # TimeLimit wrapper double-enforcing a second, possibly mismatched
    # horizon on top of it. max_episode_steps=None skips adding that extra
    # TimeLimit wrapper.
    max_episode_steps=None,
)


if __name__ == "__main__":
    # Quick smoke test, same shape as the __main__ block in
    # amzedex_cube_env2.py, but going through gym.make instead of
    # instantiating the class directly.
    env = gym.make(
        "AmazeDex/CubeGrasp-v0",
        render_mode="human",
        total_training_steps=1_000_000,
    )

    for episode in range(100):
        obs, info = env.reset(seed=episode)
        total_reward = 0.0
        terminated = truncated = False
        steps = 0

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        reason = "dropped" if info.get("dropped") else ("success" if info.get("success") else "max steps reached")
        print(
            f"episode {episode}: return={total_reward:.2f}, steps={steps}, "
            f"success={info.get('success')}, ended: {reason}"
        )

    env.close()