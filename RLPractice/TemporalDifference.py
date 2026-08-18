import gymnasium as gym
import numpy as np

# Make Environment
env = gym.make("CliffWalking-v1", is_slippery=False,render_mode="ansi")
# Define Parameters
alpha = 0.2
episodes = 10000
gamma = 0.9

V = np.zeros(env.observation_space.n)

for _ in range(episodes):
    s, _ = env.reset()
    while True:
        a = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(a)
        # TD-0 Formula
        delta = (reward +( gamma *(V[next_state])) - V[s])
        V[s] = V[s] + alpha * (delta)
        s = next_state
        if terminated or truncated:
            break

print("Learned State Values:\n", V.reshape(4,12))
env.close()

