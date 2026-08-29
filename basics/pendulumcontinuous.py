import gymnasium as gym
import numpy as np
import math
import random

env = gym.make("Pendulum-v1")
num_actions = 9
actions = np.linspace(-2.0, 2.0, num_actions)

alpha = 0.05
gamma = 0.95
epsilon = 0.9
epsilon_decay = 0.999
min_epsilon = 0.005
episodes = 35000
Q = np.zeros([16, 16, 16, num_actions])


def choose_action(s, epsilon):
    if np.random.rand() < epsilon:
        return random.randint(0, num_actions - 1)
    else:
        return np.argmax(Q[s])


def discrete(state):
    cost, sint, t_dot = state
    cost_angle = (cost + 1) / 2
    sint_angle = (sint + 1) / 2
    t_dot_angular = (t_dot + 8) / 16

    cost_clip = int(np.clip(cost_angle * 16, 0, 15))
    sint_clip = int(np.clip(sint_angle * 16, 0, 15))
    t_dot_clip = int(np.clip(t_dot_angular * 16, 0, 15))
    return cost_clip, sint_clip, t_dot_clip


episode_reward = []

for episode in range(episodes):
    state, _ = env.reset()
    state = discrete(state)
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action = choose_action(state, epsilon)
        new_state, reward, terminated, truncated, _ = env.step(
            [actions[action]]
        )
        new_state = discrete(new_state)
        Q[state][action] += alpha * (
            reward + gamma * np.max(Q[new_state]) - Q[state][action]
        )
        state = new_state
        total_reward += reward
    epsilon = max(epsilon * epsilon_decay, min_epsilon)
    episode_reward.append(total_reward)

env.close()
print(episode_reward)


test_env = gym.make("Pendulum-v1", render_mode="human")
for _ in range(6):
    state, _ = test_env.reset()

    terminated, truncated = False, False
    while not (terminated or truncated):
        state = discrete(state)
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = test_env.step(
            [actions[action]]
        )
test_env.close()