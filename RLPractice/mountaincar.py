import random
from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make("MountainCar-v0")
state = env.observation_space.shape[0]
action = env.action_space.n


class Neuralnet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


gamma = 0.99
lr = 0.001
episodes = 550
batchsize = 64
buffer = 50000
replaysize = 1000
target_update_freq = 10
epsilon = 0.985
edecay = 0.998
minepsilon = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("letss go using GPU")

policy = Neuralnet(state, action).to(device)
target = Neuralnet(state, action).to(device)
target.load_state_dict(policy.state_dict())
target.eval()

optimizer = optim.Adam(policy.parameters(), lr=lr)
criterion = nn.MSELoss()
memory = deque(maxlen=buffer)

print("Training \n")

for episode in range(1, episodes + 1):
    state, _ = env.reset()
    episode_reward = 0.0
    done = False

    while not done:
        if random.random() < epsilon:
            act = env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                act = torch.argmax(policy(state_t)).item()

        next_state, reward, terminated, truncated, _ = env.step(act)
        done = terminated or truncated

        memory.append((state, act, reward, next_state, done))
        state = next_state
        episode_reward += reward

        if len(memory) >= replaysize:
            states, actions, rewards, next_states, dones = zip(
                *random.sample(memory, batchsize)
            )

            states_t = torch.FloatTensor(np.array(states)).to(device)
            actions_t = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
            dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)

            current_q = policy(states_t).gather(1, actions_t)
            with torch.no_grad():
                max_next_q = target(next_states_t).max(1, keepdim=True)[0]
                target_q = rewards_t + (gamma * max_next_q * (1 - dones_t))

            loss = criterion(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(minepsilon, epsilon * edecay)

    if episode % target_update_freq == 0:
        target.load_state_dict(policy.state_dict())
        print(
            f"Episode {episode:4d} | Reward: {episode_reward:6.1f} | Epsilon: {epsilon:.2f}"
        )

env.close()

torch.save(policy.state_dict(), "mountaincar_dqn.pth")
print("\nModel saved to mountaincar_dqn.pth!")
print("Training complete!\n")

render_env = gym.make("MountainCar-v0", render_mode="human")
policy.eval()

for ep in range(1, 10):
    state, _ = render_env.reset()
    episode_reward = 0
    done = False

    while not done:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            act = torch.argmax(policy(state_t)).item()

        state, reward, terminated, truncated, _ = render_env.step(act)
        episode_reward += reward
        done = terminated or truncated

    print(f"Rendered Episode {ep} | Score: {episode_reward:.1f}")

render_env.close()