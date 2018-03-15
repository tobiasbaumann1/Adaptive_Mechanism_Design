import numpy as np
import tensorflow as tf
import logging
logging.basicConfig(filename='Agents.log',level=logging.DEBUG)

RANDOM_SEED = 5
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

from enum import Enum, auto
class Critic_Variant(Enum):
    INDEPENDENT = auto()
    CENTRALIZED = auto()
    CENTRALIZED_APPROX = auto()

class Agent(object):
    def __init__(self, env, learning_rate=0.001, gamma = 0.95, agent_idx = 0):
        self.sess = tf.Session()
        self.env = env
        self.n_actions = env.n_actions
        self.n_features = env.n_features
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.agent_idx = agent_idx
        self.log = [] # logs action probabilities

    def choose_action(self, s):
        action_probs = self.calc_action_probs(s)
        action = np.random.choice(range(action_probs.shape[1]), p=action_probs.ravel())  # select action w.r.t the actions prob
        self.log.append(action_probs[0,1])
        return action

    def learn_at_episode_end(self):
        pass

    def get_actions_prob_log(self):
        return self.log

    def close(self):
        self.sess.close()
        tf.reset_default_graph()

    def reset(self):
        self.sess.run(tf.global_variables_initializer())


class Actor_Critic_Agent(Agent):
    def __init__(self, env, learning_rate=0.001, n_units_actor = 20, 
            n_units_critic = 20, gamma = 0.95, agent_idx = 0, 
            critic_variant = Critic_Variant.INDEPENDENT, *args):
        super().__init__(env, learning_rate, gamma, agent_idx)
        self.actor = Actor(env, n_units_actor, learning_rate, agent_idx)
        self.critic = Critic(env, n_units_critic, learning_rate, gamma, agent_idx, 
                            critic_variant)
        self.sess.run(tf.global_variables_initializer())

    def learn(self, s, a, r, s_, done = False, *args):
        if done:
            pass
        else:
            td = self.critic.learn(self.sess,s,r,s_, *args)
            self.actor.learn(self.sess,s,a,td)

    def __str__(self):
        return 'Actor_Critic_Agent_'+str(self.agent_idx)

    def calc_action_probs(self, s):
        return self.actor.calc_action_probs(self.sess,s)

    def pass_agent_list(self, agent_list):
        self.critic.pass_agent_list(agent_list)

    def get_action_prob_variable(self):
        return self.actor.actions_prob

    def get_state_variable(self):
        return self.actor.s

    def get_policy_parameters(self):
        return [self.actor.w_l1,self.actor.b_l1,self.actor.w_pi1,self.actor.b_pi1]

class Actor(object):
    def __init__(self, env, n_units = 20, learning_rate=0.001, agent_idx = 0):
        self.s = tf.placeholder(tf.float32, [1, env.n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            self.w_l1 = tf.Variable(tf.random_normal([env.n_features,n_units],stddev=0.1))
            self.b_l1 = tf.Variable(tf.random_normal([n_units],stddev=0.1))
             
            self.l1 = tf.nn.relu(tf.matmul(self.s, self.w_l1) + self.b_l1)

            self.w_pi1 = tf.Variable(tf.random_normal([n_units,env.n_actions],stddev=0.1))
            self.b_pi1 = tf.Variable(tf.random_normal([env.n_actions],stddev=0.1))
             
            self.actions_prob = tf.nn.softmax(tf.matmul(self.l1, self.w_pi1) + self.b_pi1)

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.actions_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  

        with tf.variable_scope('trainActor'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(-self.exp_v)  

    def learn(self, sess, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def calc_action_probs(self, sess, s):
        s = s[np.newaxis, :]
        probs = sess.run(self.actions_prob, {self.s: s})   # get probabilities for all actions
        return probs

class Critic(object):
    def __init__(self, env, n_units, learning_rate, gamma, agent_idx, 
                critic_variant = Critic_Variant.INDEPENDENT):
        self.critic_variant = critic_variant
        self.env = env

        self.s = tf.placeholder(tf.float32, [1, env.n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        if self.critic_variant is Critic_Variant.CENTRALIZED:
            self.act_probs = tf.placeholder(tf.float32, shape=[1, env.n_actions * env.n_players], name="act_probs")
            self.nn_inputs = tf.concat([self.s,self.act_probs],axis=1)
        else: 
            self.nn_inputs = self.s

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.nn_inputs,
                units=n_units,  # number of hidden units
                activation=tf.nn.relu,  # None
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'+str(agent_idx)
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'+str(agent_idx)
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('trainCritic'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def pass_agent_list(self, agent_list):
        self.agent_list = agent_list

    def learn(self, sess, s, r, s_, *args):
        s,s_ = s.astype(np.float32), s_.astype(np.float32)

        if self.critic_variant is Critic_Variant.CENTRALIZED:
            if args: 
                obslist = args[0]
                obs_list = args[1]
                act_probs = np.hstack([agent.calc_action_probs(obslist[idx]) for idx, agent in enumerate(self.agent_list)])
                act_probs_ = np.hstack([agent.calc_action_probs(obs_list[idx]) for idx, agent in enumerate(self.agent_list)])
            else: 
                act_probs = np.hstack([agent.calc_action_probs(s) for idx, agent in enumerate(self.agent_list)])
                act_probs_ = np.hstack([agent.calc_action_probs(s_) for idx, agent in enumerate(self.agent_list)])
            nn_inputs = np.hstack([s[np.newaxis, :], act_probs])
            nn_inputs_ = np.hstack([s_[np.newaxis, :], act_probs_])
        else:
            nn_inputs, nn_inputs_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = sess.run(self.v, {self.nn_inputs: nn_inputs_})
        td_error, _ = sess.run([self.td_error, self.train_op],
                                          {self.nn_inputs: nn_inputs, self.v_: v_, self.r: r})
        return td_error

class Simple_Agent(Agent): #plays games with 2 actions, using a single parameter
    def __init__(self, env, learning_rate=0.001, n_units_critic = 20, gamma = 0.95, agent_idx = 0, critic_variant = Critic_Variant.INDEPENDENT):
        super().__init__(env, learning_rate, gamma, agent_idx)
        self.s = tf.placeholder(tf.float32, [1, env.n_features], "state") # dummy variable
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            self.theta = tf.Variable(tf.random_uniform([1],minval=0,maxval=1))  # theta represents probability to play action 1 (cooperate)
            self.actions_prob = tf.expand_dims(tf.concat([1-tf.sigmoid(self.theta),tf.sigmoid(self.theta)],0),0)

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.actions_prob[0,self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  

        with tf.variable_scope('trainActor'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(-self.exp_v) 

        self.critic = Critic(env, n_units_critic, learning_rate, gamma, agent_idx, critic_variant)

        self.sess.run(tf.global_variables_initializer())

    def learn(self, s, a, r, s_, done = False, *args):
        if done:
            pass
        else:
            td = self.critic.learn(self.sess,s,r,s_, *args)
            feed_dict = {self.a: a, self.td_error: td}
            _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)

    def __str__(self):
        return 'Simple_Agent_'+str(self.agent_idx)

    def calc_action_probs(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.actions_prob)  
        return probs

    def pass_agent_list(self, agent_list):
        self.critic.pass_agent_list(agent_list)

    def get_action_prob_variable(self):
        return self.actions_prob

    def get_state_variable(self):
        return self.s

    def get_policy_parameters(self):
        return [self.theta]

