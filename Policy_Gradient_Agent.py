import numpy as np
import tensorflow as tf

random_seed = 28
np.random.seed(random_seed)
tf.set_random_seed(random_seed)

class Policy_Gradient_Agent:
    def __init__(
            self,
            n_actions,
            n_features,
            n_nodes,
            learning_rate=0.01,
            reward_decay=0.95,
            agent_idx = 0
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.n_nodes = n_nodes
        self.agent_idx = agent_idx

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        self.cost_history = []

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'+str(self.agent_idx)):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # Intermediate layer
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=self.n_nodes,
            activation=tf.nn.relu, 
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='layer1_'+str(self.agent_idx)
        )
        # Last layer
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='layer2_'+str(self.agent_idx)
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def reset(self):
        #tf.reset_default_graph()
        self.sess.run(tf.global_variables_initializer())

    def store_transition(self, s, a, r, s_):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def learn(self, observation, action, reward, observation_, done=False):
        if done:
            # discount and normalize episode reward
            discounted_ep_rs_norm = self._discount_and_norm_rewards()

            # train on episode
            self.sess.run(self.train_op, feed_dict={
                 self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
                 self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
                 self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
            })

            self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        else:
            self.store_transition(observation, action, reward, observation_)    

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs = discounted_ep_rs - np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def toString(self):
        return 'Agent'+str(self.agent_idx)