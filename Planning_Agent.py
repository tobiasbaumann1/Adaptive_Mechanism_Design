import tensorflow as tf
import numpy as np
import logging
logging.basicConfig(filename='Planning_Agent.log',level=logging.DEBUG,filemode='w')
from Agents import Agent

RANDOM_SEED = 5
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)

class Planning_Agent(Agent):
    def __init__(self, env, underlying_agents, learning_rate=0.01,
        gamma = 0.95, max_reward_strength = None, cost_param = 0, with_redistribution = False, 
        proxy_value_function = False):
        super().__init__(env, learning_rate, gamma)     
        self.underlying_agents = underlying_agents
        self.log = []
        self.max_reward_strength = max_reward_strength
        n_players = len(underlying_agents)
        self.with_redistribution = with_redistribution

        self.s = tf.placeholder(tf.float32, [1, env.n_features], "state")   
        self.a_players = tf.placeholder(tf.float32, [1, n_players], "player_actions")
        self.r_players = tf.placeholder(tf.float32, [1, n_players], "player_rewards")
        self.inputs = tf.concat([self.s,self.a_players],1)

        with tf.variable_scope('Policy_p'):
            self.l1 = tf.layers.dense(
                inputs=self.inputs,
                units=n_players,    # 1 output per agent
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0, .1),  # weights
                bias_initializer=tf.constant_initializer(0),  # biases
                name='actions_planning'
            )

            if max_reward_strength is None:
                self.action_layer = self.l1
            else:
                self.action_layer = tf.sigmoid(self.l1)

        with tf.variable_scope('Vp'):
            # Vp is trivial to calculate in this special case
            if max_reward_strength is not None:
                self.vp = 2 * max_reward_strength * (self.action_layer - 0.5)
            else:
                self.vp = self.action_layer

        with tf.variable_scope('V_total'):
            if proxy_value_function:
                self.v = 2 * self.a_players - 1
            else:
                self.v = tf.reduce_sum(self.r_players) - 1.9
        with tf.variable_scope('cost_function'):
            self.g_log_pi = tf.placeholder(tf.float32, [1, n_players], "player_gradients")
            cost_list = []
            for underlying_agent in underlying_agents:
                # policy gradient theorem
                idx = underlying_agent.agent_idx
                self.g_Vp_d = self.g_log_pi[0,idx] * self.vp[0,idx]
                self.g_V_d = self.g_log_pi[0,idx] * (self.v[0,idx] if proxy_value_function else self.v)

                #cost_list.append(- underlying_agent.learning_rate * tf.tensordot(self.g_Vp_d,self.g_V_d,1))
                cost_list.append(- underlying_agent.learning_rate * self.g_Vp_d * self.g_V_d)

            if with_redistribution:
                extra_loss = cost_param * tf.norm(self.vp-tf.reduce_mean(self.vp))
            else:
                extra_loss = cost_param * tf.norm(self.vp)
            self.loss = tf.reduce_sum(tf.stack(cost_list)) + extra_loss

        with tf.variable_scope('trainPlanningAgent'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, 
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Policy_p'))  

        self.sess.run(tf.global_variables_initializer())

    def learn(self, s, a_players):
        s = s[np.newaxis,:]
        r_players = np.asarray(self.env.calculate_payoffs(a_players))
        a_players = np.asarray(a_players)
        g_log_pi_list = []
        for underlying_agent in self.underlying_agents:
            idx = underlying_agent.agent_idx
            g_log_pi_list.append(underlying_agent.calc_g_log_pi(s,a_players[idx]))
        g_log_pi_arr = np.reshape(np.asarray(g_log_pi_list),[1,-1])
        #_,g_log_pi = self.sess.run([self.train_op,self.underlying_agents[-1].g_log_pi], feed_dict)
        feed_dict = {self.s: s, self.a_players: a_players[np.newaxis,:], 
                    self.r_players: r_players[np.newaxis,:],self.g_log_pi: g_log_pi_arr}
        self.sess.run([self.train_op], feed_dict)
        #print(g_log_pi)
        action,vp,v,loss,g_Vp_d,g_V_d = self.sess.run([self.action_layer,self.vp,self.v,self.loss,
            self.g_Vp_d,self.g_V_d], feed_dict)
        logging.info('Learning step')
        logging.info('Planning_action: ' + str(action))
        logging.info('Vp: ' + str(vp))
        logging.info('V: ' + str(v))
        logging.info('Gradient of V_p: ' + str(g_Vp_d))
        logging.info('Gradient of V: ' + str(g_V_d))
        logging.info('Loss: ' + str(loss))

    def get_log(self):
        return self.log

    def choose_action(self, s, a_players):
        logging.info('Player actions: ' + str(a_players))
        s = s[np.newaxis, :]
        a_players = np.asarray(a_players)        
        a_plan = self.sess.run(self.action_layer, {self.s: s, self.a_players: a_players[np.newaxis,:]})[0,:]
        if self.max_reward_strength is not None:
            a_plan = 2 * self.max_reward_strength * (a_plan - 0.5)
        logging.info('Planning action: ' + str(a_plan))
        # Planning actions in each of the 4 cases: DD, CD, DC, CC
        a_plan_DD = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([0,0])[np.newaxis,:]})
        a_plan_CD = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([1,0])[np.newaxis,:]})
        a_plan_DC = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([0,1])[np.newaxis,:]})
        a_plan_CC = self.sess.run(self.action_layer, {self.s: s, self.a_players: np.array([1,1])[np.newaxis,:]})
        l_temp = [a_plan_DD,a_plan_CD,a_plan_DC,a_plan_CC]
        if self.max_reward_strength is not None:
            l = [2 * self.max_reward_strength * (a_plan_X[0,0]-0.5) for a_plan_X in l_temp]
        else:
            l = [a_plan_X[0,0] for a_plan_X in l_temp]
        if self.with_redistribution:
            if self.max_reward_strength is not None:
                l2 = [2 * self.max_reward_strength * (a_plan_X[0,1]-0.5) for a_plan_X in l_temp]
            else:
                l2 = [a_plan_X[0,1] for a_plan_X in l_temp]
            l = [0.5 * (elt[0]-elt[1]) for elt in zip(l,l2)] 
        # else:
        self.log.append(l)
        return a_plan