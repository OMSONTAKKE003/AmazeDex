import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', is_slippery=False,render_mode='human')
env = env.unwrapped
env.nS = env.observation_space.n
env.nA = env.action_space.n

def valueiteration(env, theta=0.001, discount_factor=0.99):
    def one_step_lookahead(state, V):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
        
        if delta < theta:
            break

    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0
        
    return policy, V

policy, V = valueiteration(env, theta=0.001, discount_factor=0.99)

print("\nValue function (reshaped to grid):")
print(V.reshape(4, 4))

action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']
policy_actions = np.argmax(policy, axis=1)
policy_grid = np.array([action_names[a][0] for a in policy_actions]).reshape(4, 4)
print("\nPolicy (first letter of best action per state):")
print(policy_grid)

n_episodes = 1000
successes = 0
for _ in range(n_episodes):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(policy[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    if reward == 1:
        successes += 1

print(f"\nSuccess rate over {n_episodes} episodes: {successes / n_episodes * 100:.1f}%")