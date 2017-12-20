import numpy as np

class IPD:
    def __init__(self, HISTORY_LENGTH, N_EPISODES):
        self.action_space = ['0', '1'] # 0 means defect, +1 cooperate
        self.n_actions = len(self.action_space)
        self.n_episodes = N_EPISODES
        self.step_ctr = 0
        self.ep_ctr = 0
        self.HISTORY_LENGTH = HISTORY_LENGTH
        self.payoff_matrix = np.array([[(1,1),(3,0)],[(0,3),(2,2)]]) #standard PD payoffs
        self.reset()

    def get_n_steps(self):
        return self.step_ctr

    def step(self, actions, curriculum):
        assert(len(actions)) == 2        
        self.step_ctr += 1

        rewards = self.calculate_payoffs(actions, curriculum)
        #shift state to the left and append actions taken
        self.s[:-1,:] = self.s[1:,:]
        self.s[-1,:] = actions
        
        #end game after 100 steps
        if self.step_ctr >= 100:
            done = True
        else:
            done = False

        return self.state_to_observation(), rewards, done

    def reset(self):
        self.s = -np.ones((self.HISTORY_LENGTH,2)) #-1 means no action (at start of game)
        self.step_ctr = 0
        self.ep_ctr += 1
        print(2*(1-self.ep_ctr/self.n_episodes))
        return self.state_to_observation()

    def reset_ep_ctr(self):
        self.ep_ctr = 0

    def state_to_observation(self):
        return np.reshape(self.s,2*self.HISTORY_LENGTH)

    def calculate_payoffs(self, actions, curriculum):
        #print(self.s)
        #print(actions)
        if curriculum and all([a == 1 for a in actions]):
            bonus = 2*max((1-self.ep_ctr/self.n_episodes),0)
            return self.payoff_matrix[actions] + bonus
        else:
            return self.payoff_matrix[actions]