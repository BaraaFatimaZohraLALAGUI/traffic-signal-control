import numpy as np
import agent as ag
import sumoenv as se
import matplotlib.pyplot as plt

env_train = se.SumoEnv(gui_f=False)
env_test = se.SumoEnv(gui_f=True)
agent = ag.Agent()

EPS = 2

cumulative_rewards = []

for ieps in range(EPS):
    episode_rewards = []
    for i in range(20):
        state = env_train.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.policy(state)
            next_state, reward, done, rewards = env_train.step_d(action)

            agent.train(state, action, reward, 0.001, [1, 1, done, 1, 1])

            state = next_state
            total_reward += reward
        episode_rewards.append(total_reward)
        env_train.close()

    cumulative_rewards.extend(episode_rewards)

    state = env_test.reset()
    done = False
    while not done:
        action = agent.policy(state)
        next_state, reward, done, rewards = env_test.step_d(action)
        print(state)
        state = next_state
    env_test.close()

# Plot cumulative rewards
plt.plot(cumulative_rewards)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Rewards per Episode')
plt.show()

