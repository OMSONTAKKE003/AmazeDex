import random
from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make("Acrobot-v1", render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


class Neural(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


episodes = 2500
gamma = 0.999
batch_size = 64
lr = 0.001
buffer_capacity = 50000
min_replay_size = 1000
target_update_freq = 10
epsilon = 0.985
epsilon_decay = 0.999
epsilon_min = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = Neural(state_dim, action_dim).to(device)
target_net = Neural(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.SGD(policy_net.parameters(), lr=lr)
memory = deque(maxlen=buffer_capacity)

print(f"Training DQN on {device}...")

for episode in range(1, episodes + 1):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = torch.argmax(policy_net(state_t)).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.append((state, action, reward, next_state, done))
        state = next_state
        episode_reward += reward

        if len(memory) >= min_replay_size:
            states, actions, rewards, next_states, dones = zip(
                *random.sample(memory, batch_size)
            )

            states_t = torch.FloatTensor(np.array(states)).to(device)
            actions_t = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
            dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)

            current_q = policy_net(states_t).gather(1, actions_t)
            with torch.no_grad():
                max_next_q = target_net(next_states_t).max(1, keepdim=True)[0]
                target_q = rewards_t + (gamma * max_next_q * (1 - dones_t))

            loss = nn.MSELoss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print(
            f"Episode {episode:3d} | Reward: {episode_reward:6.1f} | Epsilon: {epsilon:.2f}"
        )

torch.save(policy_net.state_dict(), "acrobot_dqn.pth")
print("Model saved to acrobot_dqn.pth!")
env.close()
print("Training complete!\n")

render_env = gym.make("Acrobot-v1", render_mode="human")
policy_net.eval()

for episode in range(1, 188):
    state, _ = render_env.reset()
    episode_reward = 0
    done = False

    while not done:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = torch.argmax(policy_net(state_t)).item()

        state, reward, terminated, truncated, _ = render_env.step(action)
        episode_reward += reward
        done = terminated or truncated

    print(f"Rendered Episode {episode} | Score: {episode_reward:.1f}")

render_env.close()