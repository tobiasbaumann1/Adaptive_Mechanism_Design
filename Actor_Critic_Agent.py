"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

The cart pole example. Policy is oscillated.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import numpy as np
import tensorflow as tf

np.random.seed(3)
tf.set_random_seed(3)  # reproducible

class Actor_Critic_Agent(object):
    def __init__(self, n_actions, n_features, learning_rate=0.001, n_units_actor = 20, 
            n_units_critic = 20, gamma = 0.95, agent_idx = 0):
        sess = tf.Session()
        self.actor = Actor(sess, n_features, n_actions, learning_rate, n_units_actor, agent_idx)
        self.critic = Critic(sess, n_features, learning_rate, n_units_critic, gamma, agent_idx)
        self.agent_idx = agent_idx
        sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.actor.choose_action(s)

    def learn(self, s, a, r, s_, done=False):
        if done:
            pass
        else:
            td = self.critic.learn(s,r,s_)
            self.actor.learn(s,a,td)

    def toString(self):
        return 'ActorCriticAgent_'+str(self.agent_idx)

class Actor(object):
    def __init__(self, sess, n_features, n_actions, learning_rate=0.001, n_units = 20, agent_idx = 0):
        self.sess = sess
        self.agent_idx = agent_idx

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=n_units,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'+str(self.agent_idx)
            )

            self.actions_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='actions_prob'+str(self.agent_idx)
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.actions_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  

        with tf.variable_scope('trainActor'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(-self.exp_v)  

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.actions_prob, {self.s: s})   # get probabilities for all actions
        if np.isnan(probs).any():
            print(probs)
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, learning_rate=0.01, n_units = 20, gamma = 0.95, agent_idx = 0):
        self.sess = sess
        self.agent_idx = agent_idx
        self.gamma = gamma

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=n_units,  # number of hidden units
                activation=tf.nn.relu,  # None
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'+str(self.agent_idx)
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'+str(self.agent_idx)
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + self.gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('trainCritic'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error