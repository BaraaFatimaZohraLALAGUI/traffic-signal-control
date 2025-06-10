import multiprocessing
import numpy as np
import tensorflow.compat.v1 as tf
import sumoenv as se
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

class GlobalNet:
    def __init__(self, state_dim, action_dim, scope='global'):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [None, state_dim], 'state')
            dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(self.state)
            self.policy_logits = tf.keras.layers.Dense(action_dim)(dense1)
            self.policy = tf.nn.softmax(self.policy_logits)
            self.value = tf.keras.layers.Dense(1)(dense1)

class Worker:
    def __init__(self, name, global_net, sess, env_fn, state_dim, action_dim, gamma=0.99):
        self.name = name
        self.global_net = global_net
        self.sess = sess
        self.env = env_fn(label=name)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        with tf.variable_scope(self.name):
            self.state = tf.placeholder(tf.float32, [None, state_dim], 'state')
            dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(self.state)
            self.policy_logits = tf.keras.layers.Dense(action_dim)(dense1)
            self.policy = tf.nn.softmax(self.policy_logits)
            self.value = tf.keras.layers.Dense(1)(dense1)
            self.actions = tf.placeholder(tf.int32, [None], 'actions')
            self.advantages = tf.placeholder(tf.float32, [None], 'advantages')
            self.returns = tf.placeholder(tf.float32, [None], 'returns')
            action_onehot = tf.one_hot(self.actions, action_dim, dtype=tf.float32)
            responsible_outputs = tf.reduce_sum(self.policy * action_onehot, axis=1)
            policy_loss = -tf.reduce_mean(tf.log(responsible_outputs + 1e-8) * self.advantages)
            value_loss = tf.losses.mean_squared_error(self.returns, tf.squeeze(self.value))
            self.loss = policy_loss + 0.5 * value_loss
            self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            self.global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global')
            self.grads = tf.gradients(self.loss, self.local_vars)
            optimizer = tf.train.AdamOptimizer(1e-4)
            self.apply_grads = optimizer.apply_gradients(zip(self.grads, self.global_vars))
            self.sync = [l.assign(g) for l, g in zip(self.local_vars, self.global_vars)]

    def work(self, max_episodes=10):
        self.sess.run(self.sync)
        episode_rewards = []
        for ep in range(max_episodes):
            state = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            done = False
            while not done:
                policy = self.sess.run(self.policy, {self.state: [state]})[0]
                action = np.random.choice(self.action_dim, p=policy)
                next_state, reward, done, _ = self.env.step_d(action)
                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(reward)
                state = next_state
                ep_r += reward
                if done or len(buffer_s) >= 10:
                    v_s_ = 0 if done else self.sess.run(self.value, {self.state: [next_state]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + self.gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()
                    feed_dict = {
                        self.state: buffer_s,
                        self.actions: buffer_a,
                        self.advantages: np.array(buffer_v_target) - self.sess.run(self.value, {self.state: buffer_s}).flatten(),
                        self.returns: buffer_v_target
                    }
                    self.sess.run(self.apply_grads, feed_dict)
                    self.sess.run(self.sync)
                    buffer_s, buffer_a, buffer_r = [], [], []
            print(f'{self.name} Episode {ep}, Reward: {ep_r}')
            episode_rewards.append(ep_r)
            self.env.close()
        return episode_rewards

def worker_process(worker_id, max_episodes, state_dim, action_dim, reward_dict, save_path):
    import tensorflow.compat.v1 as tf
    import sumoenv as se
    tf.disable_v2_behavior()
    sess = tf.InteractiveSession()
    # Always use non-GUI for all workers during training
    use_gui = False
    global_net = GlobalNet(state_dim, action_dim)
    worker = Worker(f'worker_{worker_id}', global_net, sess, lambda label: se.SumoEnv(label=label, gui_f=use_gui), state_dim, action_dim)
    sess.run(tf.global_variables_initializer())
    rewards = worker.work(max_episodes)
    reward_dict[worker_id] = rewards
    # Save the model from worker 0 after training
    if worker_id == 0:
        saver = tf.train.Saver()
        saver.save(sess, save_path)
    sess.close()

def run_a3c(num_workers=4, max_episodes=10, save_path='a3c_model.ckpt'):
    state_dim = 124  # 10*12+4
    action_dim = 4
    tf.reset_default_graph()
    manager = multiprocessing.Manager()
    reward_dict = manager.dict()  # Shared dict to collect rewards from workers
    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(target=worker_process, args=(i, max_episodes, state_dim, action_dim, reward_dict, save_path))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    # Gather rewards
    all_rewards = []
    for i in range(num_workers):
        all_rewards.extend(reward_dict.get(i, []))
    return all_rewards

# Add a function to run a test/visualization episode using the saved policy
def run_visualization_with_policy(save_path='a3c_model.ckpt'):
    state_dim = 124
    action_dim = 4
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    global_net = GlobalNet(state_dim, action_dim)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_path)
    import sumoenv as se
    env = se.SumoEnv(gui_f=True)
    state = env.reset()
    done = False
    while not done:
        policy = sess.run(global_net.policy, {global_net.state: [state]})[0]
        action = np.random.choice(action_dim, p=policy)
        state, reward, done, _ = env.step_d(action)
    env.close()
    sess.close()

if __name__ == '__main__':
    # 1. Train
    rewards = run_a3c(num_workers=3, max_episodes=4, save_path='a3c_model.ckpt')
    # 2. Visualize in SUMO GUI with trained policy
    print('Training complete. Launching SUMO GUI simulation with trained policy...')
    import time
    time.sleep(2)
    run_visualization_with_policy(save_path='a3c_model.ckpt')
    # 3. Plot the cumulative reward graph after SUMO GUI is closed
    plt.figure()
    plt.plot(np.arange(1, len(rewards)+1), rewards, label='Cumulative Reward per Episode')
    plt.xlabel('Training Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('A3C Training: Cumulative Reward per Episode')
    plt.legend()
    plt.show()
