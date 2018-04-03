import tensorflow as tf
import numpy as np
import logging
logging.basicConfig(filename='Policing_Agent.log',level=logging.DEBUG)
from Agents import Agent

class Policing_Agent(Agent):
    def __init__(self, env, policed_agents, learning_rate=0.01, n_units = 4, gamma = 0.95):
        super().__init__(env, learning_rate, gamma)     
        self.policing_subagents = []
        for policed_agent in policed_agents:
            self.policing_subagents.append(Policing_Sub_Agent(env,policed_agent,learning_rate,n_units,gamma))
        self.log = [] # logs action probabilities

    def learn(self, s, a_players):
        for (a,policing_subagent) in zip(a_players,self.policing_subagents):
            policing_subagent.learn(s,a)

    def choose_action(self, s, player_actions):
        s = s[np.newaxis,:]
        # TODO
        self.sess.run(feed_dict = {self.s: s, self.a_player: 0})
        {self.s: s, self.a_player: 1}
        self.log.append()
        return [policing_subagent.choose_action(s,a) for (a,policing_subagent) in zip(player_actions,self.policing_subagents)]

class Policing_Sub_Agent(Agent):
    def __init__(self, env, policed_agent, learning_rate=0.01, n_units = 4, gamma = 0.95):
        super().__init__(env, learning_rate, gamma)
        self.n_policing_actions = 2
        self.n_features = env.n_features + env.n_actions * env.n_players
        self.policed_agent = policed_agent

        self.s = tf.placeholder(tf.float32, [1, env.n_features], "state")   
        self.a_player = tf.placeholder(tf.float32, None, "player_action")
        self.inputs = tf.concat([self.s,tf.reshape(self.a_player,(1,1))],1)

        with tf.variable_scope('Policy_p_'+str(policed_agent.agent_idx)):
            # l1 = tf.layers.dense(
            #     inputs=self.inputs,
            #     units=n_units,    # number of hidden units
            #     activation=tf.nn.relu,
            #     kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
            #     bias_initializer=tf.constant_initializer(0),  # biases
            #     name='l1_policing'
            # )

            self.action_probs = tf.layers.dense(
                inputs=self.inputs,
                units=self.n_policing_actions,    # output units
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0),  # biases
                name='actions_policing'
            )

        with tf.variable_scope('Vp'):
            # Vp is trivial to calculate in this special case
            self.vp = 3 * (self.action_probs[0,1]-self.action_probs[0,0])

        with tf.variable_scope('V_total'):
            # V is trivial to calculate in this special case
            self.v = 4 * (self.a_player - 0.5) # omit contribution of second player because derivative vanishes

        with tf.variable_scope('cost_function'):
            # Gradients w.r.t. theta_1
            log_prob_pi = tf.log(policed_agent.get_action_prob_variable()[0,tf.cast(self.a_player,dtype = tf.int32)])
            theta = policed_agent.get_policy_parameters()
            g_log_prob = [tf.gradients(log_prob_pi,param) for param in theta]
            g_log_prob = tf.concat([tf.reshape(param,[-1]) for param in g_log_prob],0)

            # policy gradient theorem
            self.g_Vp_d = g_log_prob * self.vp
            self.g_V_d = g_log_prob * self.v

            self.cost = - policed_agent.learning_rate * tf.tensordot(self.g_Vp_d,self.g_V_d,1)

        with tf.variable_scope('trainPolicingAgent'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost, 
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Policy_p_'+str(policed_agent.agent_idx)))  

        self.sess.run(tf.global_variables_initializer())

    def learn(self, s, a_player):
        s = s[np.newaxis,:]
        feed_dict = {self.s: s, self.a_player: a_player, self.policed_agent.get_state_variable(): s}
        self.sess.run([self.train_op], feed_dict)
        action_probs,vp,v,cost,g_Vp_d,g_V_d = self.sess.run([self.action_probs,self.vp,self.v,self.cost,self.g_Vp_d,self.g_V_d], feed_dict)
        logging.info('Policing_action_probs: ' + str(action_probs))
        logging.info('Vp: ' + str(vp))
        logging.info('V: ' + str(v))
        logging.info('Gradient of V_p: ' + str(g_Vp_d))
        logging.info('Gradient of V: ' + str(g_V_d))
        logging.info('Cost: ' + str(cost))

    def choose_action(self, s, a):
        action_probs = self.calc_action_probs(s,a)
        logging.info('Player action: ' + str(a))
        action = np.random.choice(range(action_probs.shape[1]), p=action_probs.ravel())  # select action w.r.t the actions prob
        logging.info('Policing action: ' + str(action))
        return action

    def calc_action_probs(self, s, a):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.action_probs, {self.s: s, self.a_player: a})   # get probabilities for all actions
        return probs