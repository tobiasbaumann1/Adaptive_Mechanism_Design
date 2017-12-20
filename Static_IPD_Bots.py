class TitForTatBot:
    def __init__(
            self,
            player_index
    ):
        self.player_index = player_index
        assert player_index == 0 or player_index == 1

    def choose_action(self, observation):
        if self.player_index == 0:
            last_opponent_action = int(observation[-1])
        else:
            last_opponent_action = int(observation[-2])
        #Cooperate at the start
        if last_opponent_action == -1:
            return 1
        #Copy last opponent action
        else:
            return last_opponent_action

    def learn(self, *args):
        pass

class CooperateBot: 
    def choose_action(self, _):
        return 1 #always cooperate

    def learn(self, *args):
        pass

class DefectBot: 
    def choose_action(self, _):
        return 0 #always defect

    def learn(self, *args):
        pass