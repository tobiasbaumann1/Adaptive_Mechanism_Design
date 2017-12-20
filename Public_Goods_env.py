import numpy as np

class Public_Goods_Game:
    def __init__(self, HISTORY_LENGTH, N_EPISODES, N_PLAYERS, 
            multiplier = 2, punishment_cost = 0.2, punishment_strength = 1):
        self.n_actions = 3
        self.n_episodes = N_EPISODES
        self.n_players = N_PLAYERS
        self.step_ctr = 0
        self.ep_ctr = 0
        self.multiplier = multiplier
        self.punishment_cost = punishment_cost
        self.punishment_strength = punishment_strength
        self.HISTORY_LENGTH = HISTORY_LENGTH
        self.reset()

    def get_n_steps(self):
        return self.step_ctr

    def step(self, actions, curriculum):
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
        self.s = -np.ones((self.HISTORY_LENGTH,self.n_players)) #-1 means no action (at start of game)
        self.step_ctr = 0
        self.ep_ctr += 1
        return self.state_to_observation()

    def reset_ep_ctr(self):
        self.ep_ctr = 0

    def state_to_observation(self):
        return np.reshape(self.s,self.n_players*self.HISTORY_LENGTH)

    def calculate_payoffs(self, actions, curriculum):
        # if curriculum and all([a == 1 for a in actions]):
        #     bonus = max((1-self.ep_ctr/self.n_episodes),0)
        #     return self.payoff_matrix[actions] + bonus
        # else:
        totalPool = self.multiplier * sum([min(a,1) for a in actions])
        share = totalPool / self.n_players
        payoffs = [share - min(a,1) for a in actions] # before punishment
        punishment_costs = [self.punishment_cost if a == 2 else 0 for a in actions]
        number_of_freeriders = actions.count(0)
        if number_of_freeriders > 0:
            punishment_amount_per_freerider = self.punishment_strength * actions.count(2) * 1.0 / number_of_freeriders
            punishments = [punishment_amount_per_freerider if a == 0 else 0 for a in actions]
        else:
            punishments = [0] * self.n_players
        return [r1 - r2 - r3
                for r1,r2,r3 in zip(payoffs,punishment_costs,punishments)]