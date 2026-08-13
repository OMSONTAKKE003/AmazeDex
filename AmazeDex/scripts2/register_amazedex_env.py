import gymnasium as gym
import numpy as np

from amazedex_cube_env import AmazeDexCubeEnv

gym.register(
    id="AmazeDex/CubeRotate-v0",
    entry_point=AmazeDexCubeEnv,
    max_episode_steps=1000,
)


if __name__ == "__main__":
    env = gym.make("AmazeDex/CubeRotate-v0", render_mode="human")
    
    total_transition_steps = 1000

    for episode in range(100):
        obs, info = env.reset(seed=episode)
        total_reward = 0.0
        terminated = truncated = False
        steps = 0
        low = env.action_space.low
        high = env.action_space.high
        
        while not (terminated or truncated):
            alpha = min(steps / total_transition_steps, 1.0)
            
            action = (1 - alpha) * high + (alpha * low)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
        reason = "cube dropped/lost from frame" if terminated else "max steps reached"
        print(f"episode {episode}: return={total_reward:.2f}, steps={steps}, ended: {reason}")

    env.close()