import matplotlib.pyplot as plt
#from IPD_env import IPD
from Environments import Public_Goods_Game
# from DQN_Agent import DQN_Agent
#from Policy_Gradient_Agent import Policy_Gradient_Agent
from Agents import Actor_Critic_Agent
#from Static_IPD_Bots import *

def run_game(N_EPISODES, players):
    env.reset_ep_ctr()
    for episode in range(N_EPISODES):
        # initial observation
        s = env.reset()

        while True:
            # choose action based on s
            actions = [player.choose_action(s) for player in players]

            # take action and get next s and reward
            s_, rewards, done = env.step(actions)

            for player in players:
                player.learn(s, actions[player.agent_idx], rewards[player.agent_idx], s_)

            # swap s
            s = s_

            # break while loop when done
            if done:
                for player in players:
                    player.learn_at_episode_end() 
                break

        # end of game
        if (episode+1) % 10 == 0:
            print('Episode {} finished.'.format(episode + 1))
    return env.get_avg_rewards_per_round()

def plot_results(avg_rewards_per_round, legend):
    for idx in range(avg_rewards_per_round.shape[1]):
        plt.plot(avg_rewards_per_round[:,idx])
    plt.xlabel('Episode')
    plt.ylabel('Average reward per round')
    plt.legend(legend)
    plt.show()

if __name__ == "__main__":
    HISTORY_LENGTH = 0 # the NN will use the actions from this many past rounds to determine its action
    N_EPISODES = 100
    N_PLAYERS = 4
    N_UNITS = 16 #number of nodes in the intermediate layer of the NN
    # Initialize env and agents
    env = Public_Goods_Game(HISTORY_LENGTH, N_EPISODES,N_PLAYERS, multiplier = 3, punishment_cost = 0.1, punishment_strength = 2)    
    # agent = DQN_Agent(env.n_actions, 2*HISTORY_LENGTH, N_UNITS,
    #                   learning_rate=0.1,
    #                   reward_decay=0.9,
    #                   e_greedy=0.9,
    #                   replace_target_iter=20,
    #                   memory_size=2000,
    #                   )
    agents = [Actor_Critic_Agent(env.n_actions, N_PLAYERS*HISTORY_LENGTH, 
                      learning_rate=0.001,
                      gamma=0.9,
                      agent_idx = i) for i in range(N_PLAYERS)]

    avg_rewards_per_round = run_game(N_EPISODES,agents)
    plot_results(avg_rewards_per_round,[str(agent) for agent in agents])
    # PG_agent0.reset()
    # PG_agent1.reset()
    # avg_rewards = run_game(N_EPISODES,True,agent_list)
    # plot_results(avg_rewards,['Agent0','Agent1']), agent_idx = 0