import numpy as np
import agent as ag
import sumoenv as se
import matplotlib.pyplot as plt

env_train = se.SumoEnv(gui_f=False)
env_test = se.SumoEnv(gui_f=True)
agent = ag.A2CAgent()

EPS = 2
train_rewards = []
test_rewards = []

for ieps in range(EPS):
    for i in range(20):
        state = env_train.reset()
        done = False
        ep_r = 0
        while not done:
            action = agent.policy(state)
            next_state, reward, done, rewards = env_train.step_d(action)
            agent.train(state, action, reward, 0.001, [1, 1, done, 1, 1])
            state = next_state
            ep_r += reward
        train_rewards.append(ep_r)
        env_train.close()

    state = env_test.reset()
    done = False
    ep_r = 0
    while not done:
        action = agent.policy(state)
        next_state, reward, done, rewards = env_test.step_d(action)
        print(state)
        state = next_state
        ep_r += reward
    test_rewards.append(ep_r)
    env_test.close()

# Plot training rewards only
plt.figure()
plt.plot(np.arange(1, len(train_rewards)+1), train_rewards, label='Cumulative Reward per Train Episode')
plt.xlabel('Training Episode')
plt.ylabel('Cumulative Reward')
plt.title('A2C Training: Cumulative Reward per Episode')
plt.legend()
plt.show()

