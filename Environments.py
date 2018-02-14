import numpy as np

class Environment(object):
    def __init__(self, N_ACTIONS, N_PLAYERS, EPISODE_LENGTH, N_FEATURES = 0):
        self.n_actions = N_ACTIONS
        self.n_players = N_PLAYERS
        self.n_features = N_FEATURES
        self.episode_length = EPISODE_LENGTH
        self.step_ctr = 0
        self.ep_ctr = 0
        self.actions_list = []
        self.avg_rewards_per_round = []        
        self.reset()

    def step(self, actions):
        self.actions_list.append(actions)
        rewards = self.calculate_payoffs(actions)
        self.stored_rewards[:,self.step_ctr] = rewards
        self.update_state(actions)
        self.step_ctr += 1
        return self.state_to_observation(), rewards, self.is_done()

    def reset(self):
        self.s = self.initial_state()
        self.actions_list = []
        self.step_ctr = 0
        self.stored_rewards = np.zeros((self.n_players,self.episode_length))
        self.ep_ctr += 1
        return self.state_to_observation()

    def reset_ep_ctr(self):
        self.ep_ctr = 0

    def state_to_observation(self):
        return self.s

    def update_state(self, actions):
        pass

    def initial_state(self):
        return None

    def is_done(self):
        if self.step_ctr >= self.episode_length:
            self.avg_rewards_per_round.append(np.mean(self.stored_rewards,axis=1))
            return True
        else:
            return False

    def get_avg_rewards_per_round(self):
        return np.asarray(self.avg_rewards_per_round)

class Public_Goods_Game(Environment):
    def __init__(self, HISTORY_LENGTH, N_PLAYERS, 
            multiplier = 2, punishment_cost = 0.2, punishment_strength = 1):
        self.multiplier = multiplier
        self.punishment_cost = punishment_cost
        self.punishment_strength = punishment_strength
        self.history_length = HISTORY_LENGTH
        super().__init__(3, N_PLAYERS, 100, HISTORY_LENGTH * N_PLAYERS)

    def update_state(self, actions):
        if self.history_length > 0:
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

    def __str__(self):
        return "Public_Goods_Game"

class Multi_Agent_Random_Prisoners_Dilemma(Environment):
    def __init__(self, N_PLAYERS, rep_update_factor):
        super().__init__(2, N_PLAYERS, 100, N_PLAYERS**2+1)
        self.rep_update_factor = rep_update_factor
        self.n_coop_defect_list = []

    def update_state(self, actions):
        for idx, a in enumerate(actions):
            opp_idx = int(self.fixture[idx])
            self.s[idx,opp_idx] = (1-self.rep_update_factor) * self.s[idx,opp_idx] + self.rep_update_factor * a
            if a == 1:
                self.n_cooperate[idx,opp_idx] += 1
            else:
                self.n_defect[idx,opp_idx] += 1

    def reset(self):
        self.n_defect = np.zeros((self.n_players,self.n_players))
        self.n_cooperate = np.zeros((self.n_players,self.n_players))
        return super().reset()

    def initial_state(self):
        return np.ones((self.n_players,self.n_players))

    def state_to_observation(self):
        self.set_fixture()
        self.obs_list = [np.insert(np.reshape(self.s,self.n_players*self.n_players),0,self.fixture[i]) for i in range(self.n_players)]
        return self.obs_list

    def set_fixture(self):
        assert(self.n_players%2==0)
        fixture = np.zeros(self.n_players)
        remaining_indices = list(range(self.n_players))
        while remaining_indices:
            pair = np.random.choice(remaining_indices, 2, replace = False)
            fixture[pair[0]] = pair[1]
            fixture[pair[1]] = pair[0]
            remaining_indices.remove(pair[0])
            remaining_indices.remove(pair[1])
        self.fixture = fixture

    def is_done(self):
        self.n_coop_defect_list.append((self.n_cooperate, self.n_defect))
        return super().is_done()

    def calculate_payoffs(self, actions):
        return [1 - a + 2*actions[int(self.fixture[idx])] for idx, a in enumerate(actions)]

    def __str__(self):
        return "Prisoner's Dilemma between randomly selected agents"

class Prisoners_Dilemma(Environment):
    def __init__(self, signal_possible = False, option_to_abstain = False):
        self.signal_possible = signal_possible
        self.option_to_abstain = option_to_abstain
        N_ACTIONS = 3 if option_to_abstain else 2
        N_FEATURES = 3 if signal_possible else 1
        EPISODE_LENGTH = 2 if signal_possible else 1
        super().__init__(N_ACTIONS, 2, EPISODE_LENGTH, N_FEATURES)

    def initial_state(self):
        if self.signal_possible:
            return np.zeros(3)
        else:
            return np.zeros(1) #dummy feature to avoid errors. No meaning

    def update_state(self, actions):
        if self.signal_possible:
            if self.s[0] == 0: # 0 corresponds to being in the communication phase, 1 in the actual execution phase
                self.s[0] = 1 
                self.s[1] = min(actions[0],1)
                self.s[2] = min(actions[1],1)
        else:
            pass

    def calculate_payoffs(self, actions):
        if self.signal_possible and self.s[0] == 0:
            return [0] * 2 # no rewards in communication step
        else:
            if self.option_to_abstain and 2 in actions:
                return [0] * 2
            else:
                r0 = -1 - 2 * actions[0] + 4*actions[1] 
                r1 = -1 - 2 * actions[1] + 4*actions[0] 
                if self.signal_possible:
                    r0 = r0 - 4 * self.s[1]*(1-actions[0])
                    r1 = r1 - 4 * self.s[2]*(1-actions[1])
                return [r0,r1]

    def __str__(self):
        description = "Prisoner's_dilemma"
        if self.option_to_abstain:
            description = description + "_with_option_to_abstain" 
        return description