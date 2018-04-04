import tensorflow as tf
import numpy as np
import logging
logging.basicConfig(filename='Policing_Agent.log',level=logging.DEBUG,filemode='w')
from Agents import Agent
MAX_REWARD_STRENGTH = 3

class Policing_Agent(Agent):
    def __init__(self, env, policed_agents, learning_rate=0.01, n_units = 4, gamma = 0.95):
        super().__init__(env, learning_rate, gamma)     
        self.policing_subagents = []
        for policed_agent in policed_agents:
            self.policing_subagents.append(Policing_Sub_Agent(env,policed_agent,learning_rate,n_units,gamma))

    def learn(self, s, a_players):
        for (a,policing_subagent) in zip(a_players,self.policing_subagents):
            policing_subagent.learn(s,a)

    def get_log(self):
        return [subagent.log for subagent in self.policing_subagents]

    def choose_action(self, s, player_actions):
        return [policing_subagent.choose_action(s,a) for (a,policing_subagent) in zip(player_actions,self.policing_subagents)]

class Policing_Sub_Agent(Agent):
    def __init__(self, env, policed_agent, learning_rate=0.01, n_units = 4, gamma = 0.95):
        super().__init__(env, learning_rate, gamma)
        self.n_policing_actions = 2
        self.policed_agent = policed_agent
        self.log = [] # logs action probabilities

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

            self.l1 = tf.layers.dense(
                inputs=self.inputs,
                units=1,    # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0, .1),  # weights
                bias_initializer=tf.constant_initializer(0),  # biases
                name='actions_policing'
            )

            self.action_layer = tf.sigmoid(self.l1)

        with tf.variable_scope('Vp'):
            # Vp is trivial to calculate in this special case
            self.vp = 2 * MAX_REWARD_STRENGTH * (self.action_layer - 0.5)

        with tf.variable_scope('V_total'):
            # V is trivial to calculate in this special case
            self.v = 4 * (self.a_player - 0.5) # omit contribution of second player because derivative vanishes

        with tf.variable_scope('cost_function'):
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
        action,vp,v,cost,g_Vp_d,g_V_d = self.sess.run([self.action_layer,self.vp,self.v,self.cost,self.g_Vp_d,self.g_V_d], feed_dict)
        logging.info('Learning step')
        logging.info('Policing_action: ' + str(action))
        logging.info('Vp: ' + str(vp))
        logging.info('V: ' + str(v))
        logging.info('Gradient of V_p: ' + str(g_Vp_d))
        logging.info('Gradient of V: ' + str(g_V_d))
        logging.info('Cost: ' + str(cost))

    def choose_action(self, s, a):
        logging.info('Player action: ' + str(a))
        s = s[np.newaxis, :]
        action = self.sess.run(self.action_layer, {self.s: s, self.a_player: a}) 
        logging.info('Policing action: ' + str(action))
        # Policing action probs in the PD case
        if a == 0:
            a_p_defect = 2 * MAX_REWARD_STRENGTH * (action[0,0] - 0.5)
            a_p_coop = 2 * MAX_REWARD_STRENGTH * (self.sess.run(self.action_layer, {self.s: s, self.a_player: 1})[0,0] - 0.5)
        else:
            a_p_coop = 2 * MAX_REWARD_STRENGTH * (action[0,0] - 0.5)
            a_p_defect = 2 * MAX_REWARD_STRENGTH * (self.sess.run(self.action_layer, {self.s: s, self.a_player: 0})[0,0] - 0.5)
        self.log.append([a_p_defect,a_p_coop])
        return 2 * MAX_REWARD_STRENGTH * (action[0,0] - 0.5)

