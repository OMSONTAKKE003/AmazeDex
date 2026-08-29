import gymnasium as gym
import numpy as np
import math
import random
env = gym.make('CartPole-v1')   
env.nA = env.action_space.n
# Hyperparameters 
alpha = 0.1                
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995    
min_epsilon = 0.01
no_ep = 10000  
Q=np.zeros([10,10,10,10,2]) #10 discrete sapces and 2 discrete actions
def choose_action(s, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[s])



def discrete(state):
    cart_pos, cart_vel, pole_pos, pol_vel=state  
    cart_pos_scale=(cart_pos+4.8)/9.6  # Scale cart position to [0,1]
    cart_vel_scale=(cart_vel+3)/6  # Scale cart velocity to [0,1]
    pole_pos_scale=(pole_pos+0.418)/0.836  # Scale pole angle to [0,1]
    pol_vel_scale=(pol_vel+math.radians(50))/(2*math.radians(50))  # Scale pole angular velocity to [0,1]

    cart_pos_disc = int(np.clip(cart_pos_scale*10, 0, 9))
    cart_vel_disc = int(np.clip(cart_vel_scale*10, 0, 9))
    pole_pos_disc = int(np.clip(pole_pos_scale*10, 0, 9))
    pole_vel_disc = int(np.clip(pol_vel_scale*10, 0, 9))
    return cart_pos_disc,cart_vel_disc,pole_pos_disc,pole_vel_disc

#Training:

episode_reward = []   # move this above the for-loop, so it persists

for episode in range(no_ep):
    state, _ = env.reset()
    state = discrete(state)
    terminated = truncated = False
    total_reward = 0
    while not (terminated or truncated):
        action = choose_action(state, epsilon)
        new_state, reward, terminated, truncated, _ = env.step(action)
        new_state = discrete(new_state)
        Q[state][action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][action])
        state = new_state
        total_reward += reward
    epsilon = max(epsilon * epsilon_decay, min_epsilon)
    episode_reward.append(total_reward)   # one append per episode, list persists

env.close()
print(episode_reward)


test_env = gym.make('CartPole-v1', render_mode='human')
for _ in range(1):
    state, _ = test_env.reset()
    
    terminated, truncated = False, False
    while not (terminated or truncated):
        state=discrete(state)
        action = np.argmax(Q[state])     # epsilon=0 here - always exploit, never explore
        state, reward, terminated, truncated, _ = test_env.step(action)
test_env.close()






    
