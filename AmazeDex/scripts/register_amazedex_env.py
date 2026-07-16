"""Register AmazeDexCubeEnv so it can be created with gym.make(...).

Minimal addition on top of the existing files -- no changes needed to
mujoco_env.py or amazedex_cube_env.py. Just import this module (or run it)
before calling gym.make.
"""

import gymnasium as gym
import numpy as np

from amazedex_cube_env import AmazeDexCubeEnv

gym.register(
    id="AmazeDex/CubeRotate-v0",
    entry_point=AmazeDexCubeEnv,
    max_episode_steps=500,  # matches MAX_STEPS in amazedex_cube_env.py
)


if __name__ == "__main__":
    # Quick smoke test, same shape as the __main__ block in amazedex_cube_env.py,
    # but now going through gym.make instead of instantiating the class directly.
    env = gym.make("AmazeDex/CubeRotate-v0", render_mode="human")

    for episode in range(100):
        obs, info = env.reset(seed=episode)
        total_reward = 0.0
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            
            #action = np.zeros(8, dtype=np.float32)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        reason = "cube dropped/lost from frame" if terminated else "max steps reached"
        print(f"episode {episode}: return={total_reward:.2f}, steps={steps}, ended: {reason}")

    env.close()
    
