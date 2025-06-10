import a3c_agent as a3c
import sumoenv as se
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

def run_test_with_policy(model_path='a3c_model.ckpt'):
    state_dim = 124
    action_dim = 4
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    global_net = a3c.GlobalNet(state_dim, action_dim)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)
    env = se.SumoEnv(gui_f=True)
    state = env.reset()
    done = False
    test_rewards = []
    while not done:
        policy = sess.run(global_net.policy, {global_net.state: [state]})[0]
        action = np.argmax(policy)
        state, reward, done, _ = env.step_d(action)
        test_rewards.append(reward)
        print("State:", state)
    env.close()
    sess.close()
    # Plot the test episode rewards
    plt.plot(test_rewards)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Test Episode Rewards (A3C, Trained Policy)')
    plt.show()

if __name__ == '__main__':
    run_test_with_policy('a3c_model.ckpt')
