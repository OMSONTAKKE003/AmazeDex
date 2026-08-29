import gymnasium as gym
import numpy as np
import random

env = gym.make('Taxi-v4', is_rainy='True')   
env.nS = env.observation_space.n
env.nA = env.action_space.n

# Hyperparameters 
alpha = 0.1                
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.9995    
min_epsilon = 0.01
no_ep = 10000               

Q = np.zeros([env.nS, env.nA])

def choose_action(s, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[s, :])



for episode in range(no_ep):
    state, _ = env.reset()          
    terminated = False            
    truncated = False
    total_reward = 0

    while not (terminated or truncated):    
        action = choose_action(state, epsilon)                                    
        new_state, reward, terminated, truncated, _ = env.step(action)

       
        Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

        state = new_state
        total_reward += reward

    epsilon = max(epsilon * epsilon_decay, min_epsilon)   
    

 
    

env.close()

# ---- Test the learned policy: pure greedy, render so you can watch it drive ----
test_env = gym.make('Taxi-v4', is_rainy='False', render_mode='human')
for _ in range(10):
    state, _ = test_env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = np.argmax(Q[state, :])     # epsilon=0 here - always exploit, never explore
        state, reward, terminated, truncated, _ = test_env.step(action)
test_env.close()