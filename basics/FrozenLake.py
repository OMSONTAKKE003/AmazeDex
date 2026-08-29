import random
import gymnasium as gym
import numpy as np

#loaded the gymnasium environment for Frozen lake
environment = gym.make("FrozenLake-v1", map_name="4x4", render_mode="rgb_array", is_slippery=False)


#Set all the necessary parameters
alpha = 0.2
gamma = 0.95
epsilon = 0.8
epsilon_decay = 0.9995
min_epsilon = 0.01
num_episodes = 10000
max_steps = 200

#Make A Q-table with all zeroes
q_table = np.zeros((environment.observation_space.n, environment.action_space.n))

#Define a function to choose an action based on 
#epsilon-greedy policy to ensure exploitation and exploration are in balance
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return environment.action_space.sample()
    else:
        return np.argmax(q_table[state, :])

#for loop to run through all the episodes
for episode in range(num_episodes):
    state, _ = environment.reset()
    done = False
    
    for step in range(max_steps):
        action = choose_action(state)
        next_state, reward, done, truncated, info = environment.step(action)
        
        # Update the Q-table using the Q-learning update rule
        next_max = np.max(q_table[next_state])
        
        #TD Formula
        q_table[state, action] += alpha * (reward + gamma * next_max - q_table[state, action])
        
        state = next_state
        
        #break the loop if the episode is done or truncated
        if done or truncated:
            break
            
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
environment.close()    

#load the environment again for viewing the results of the trained agent
environment = gym.make("FrozenLake-v1", map_name="4x4", render_mode="human", is_slippery=False)


for episode in range(5):
    state, _ = environment.reset()
    done = False
    print(f"Episode {episode}")
    
    for step in range(max_steps):
        environment.render()
        action = np.argmax(q_table[state, :])
        next_state, reward, done, truncated, info = environment.step(action)
        
        state = next_state
        
        if done or truncated:
            environment.render()
            print(f"Finished episode {episode} with reward {reward}")
            break

environment.close()