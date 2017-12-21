import matplotlib.pyplot as plt
#from IPD_env import IPD
from Environments import Public_Goods_Game
# from DQN_Agent import DQN_Agent
#from Policy_Gradient_Agent import Policy_Gradient_Agent
from Agents import Actor_Critic_Agent
#from Static_IPD_Bots import *
import numpy as np

def run_game(N_EPISODES, players):
    env.reset_ep_ctr()
    avg_rewards = np.zeros((len(players),N_EPISODES))
    for episode in range(N_EPISODES):
        # initial observation
        observation = env.reset()
        rewards_sum = np.zeros(len(players))

        while True:
            # choose action based on observation
            actions = [player.choose_action(observation) for player in players]

            # take action and get next observation and reward
            observation_, rewards, done = env.step(actions)
            rewards_sum += rewards

            for player in players:
                player.learn(observation, actions[player.agent_idx], rewards[player.agent_idx], observation_)

            # swap observation
            observation = observation_

            # break while loop when done
            if done:
                for player in players:
                    player.learn(observation, actions[player.agent_idx], rewards[player.agent_idx], observation_, done)
                break

        # end of game
        if (episode+1) % 10 == 0:
            print('Episode {} done.'.format(episode + 1, env.step_ctr))
        avg_rewards[:,episode] = rewards_sum * 1.0 / env.step_ctr
    return avg_rewards

def plot_results(avg_rewards, legend):
    for idx in range(avg_rewards.shape[0]):
        plt.plot(avg_rewards[idx,:])
    plt.xlabel('Episode')
    plt.ylabel('Average reward per round')
    plt.legend(legend)
    plt.show()

if __name__ == "__main__":
    HISTORY_LENGTH = 5 # the NN will use the actions from this many past rounds to determine its action
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

    avg_rewards = run_game(N_EPISODES,agents)
    plot_results(avg_rewards,[str(agent) for agent in agents])
    # PG_agent0.reset()
    # PG_agent1.reset()
    # avg_rewards = run_game(N_EPISODES,True,agent_list)
    # plot_results(avg_rewards,['Agent0','Agent1']), agent_idx = 0