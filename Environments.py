import numpy as np

class Environment(object):
    def __init__(self, N_ACTIONS, N_PLAYERS, EPISODE_LENGTH,
            multiplier = 2, punishment_cost = 0.2, punishment_strength = 1):
        self.n_actions = N_ACTIONS
        self.n_players = N_PLAYERS
        self.episode_length = EPISODE_LENGTH
        self.step_ctr = 0
        self.ep_ctr = 0

    def step(self, actions):
        self.step_ctr += 1
        self.update_state(actions)
        rewards = self.calculate_payoffs(actions)
        return self.state_to_observation(), rewards, self.is_done()

    def update_state(self, actions):
        pass

    def calculate_payoffs(self, actions):
        pass

    def reset(self):
        self.s = self.initial_state()
        self.step_ctr = 0
        self.ep_ctr += 1
        return self.state_to_observation()

    def reset_ep_ctr(self):
        self.ep_ctr = 0

    def state_to_observation(self):
        return self.s

    def is_done(self):
        if self.step_ctr >= self.episode_length:
            return True
        else:
            return False

class Public_Goods_Game(Environment):
    def __init__(self, HISTORY_LENGTH, N_EPISODES, N_PLAYERS, 
            multiplier = 2, punishment_cost = 0.2, punishment_strength = 1):
        super().__init__(3, N_PLAYERS, 100)
        self.n_episodes = N_EPISODES
        self.multiplier = multiplier
        self.punishment_cost = punishment_cost
        self.punishment_strength = punishment_strength
        self.history_length = HISTORY_LENGTH
        self.reset()

    def update_state(self, actions):
        self.s[:-1,:] = self.s[1:,:]
        self.s[-1,:] = actions

    def initial_state(self):
        return -np.ones((self.history_length,self.n_players)) #-1 means no action (at start of game)

    def state_to_observation(self):
        return np.reshape(self.s,self.n_players*self.history_length)

    def calculate_payoffs(self, actions):
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