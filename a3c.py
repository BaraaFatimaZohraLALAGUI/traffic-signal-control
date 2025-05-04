//1. Define the Shared Brain
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
import threading

class Brain:
    def __init__(self, state_dim, action_dim, learning_rate=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)
        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()

    def _build_model(self):
        inputs = Input(shape=(self.state_dim,))
        dense = Dense(128, activation='relu')(inputs)

        policy_logits = Dense(self.action_dim, activation='softmax')(dense)
        value = Dense(1, activation='linear')(dense)

        model = Model(inputs=inputs, outputs=[policy_logits, value])
        model._make_predict_function()
        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        a_t = tf.placeholder(tf.float32, shape=(None, self.action_dim))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))

        p, v = model(s_t)

        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keepdims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = -log_prob * tf.stop_gradient(advantage)
        loss_value = 0.5 * tf.square(advantage)
        entropy = 0.01 * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keepdims=True)

        loss = tf.reduce_mean(loss_policy + loss_value - entropy)

        optimizer = tf.train.RMSPropOptimizer(1e-4)
        minimize = optimizer.minimize(loss)

        return s_t, a_t, r_t, minimize

    def train(self, s, a, r):
        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={
            s_t: s, a_t: a, r_t: r
        })

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
        return p, v


//2. Define the A3C Agent
class Agent:
    def __init__(self, brain, state_dim, action_dim):
        self.brain = brain
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.R = 0

    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        p, _ = self.brain.predict(state)
        action = np.random.choice(self.action_dim, p=p[0])
        return action

    def train(self, s, a, r, s_):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[a] = 1

        self.memory.append((s, one_hot_action, r))

        if s_ is None or len(self.memory) >= 5:
            R = 0 if s_ is None else self.brain.predict(np.expand_dims(s_, axis=0))[1][0][0]

            states, actions, rewards = [], [], []

            for s, a, r in reversed(self.memory):
                R = r + 0.99 * R
                states.append(s)
                actions.append(a)
                rewards.append([R])

            states.reverse(); actions.reverse(); rewards.reverse()
            self.brain.train(np.vstack(states), np.vstack(actions), np.vstack(rewards))
            self.memory = []

//3. Threaded Environment Runner
import threading
import time

class EnvironmentRunner(threading.Thread):
    def __init__(self, env_fn, agent, render=False):
        threading.Thread.__init__(self)
        self.env = env_fn()
        self.agent = agent
        self.render = render
        self.stop_signal = False

    def run(self):
        while not self.stop_signal:
            state = self.env.reset()
            done = False
            while not done and not self.stop_signal:
                if self.render:
                    self.env.render()
                action = self.agent.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.train(state, action, reward, None if done else next_state)
                state = next_state

    def stop(self):
        self.stop_signal = True

//4. Launch Multiple Agents
from your_env_package import TrafficEnv  # replace with your actual import

state_dim = 16     # example dimension
action_dim = 4     # number of phases per intersection

brain = Brain(state_dim, action_dim)

agents = [Agent(brain, state_dim, action_dim) for _ in range(4)]
runners = [EnvironmentRunner(lambda: TrafficEnv(), agent) for agent in agents]

for r in runners: r.start()
time.sleep(60)  # run for 60 seconds
for r in runners: r.stop()
for r in runners: r.join()

