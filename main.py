import matplotlib.pyplot as plt
from IPD_env import IPD
# from DQN_Agent import DQN_Agent
from Policy_Gradient_Agent import Policy_Gradient_Agent
from Static_IPD_Bots import *
import numpy as np

def run_game(N_EPISODES, *players):
    avg_rewards = np.zeros((len(players),N_EPISODES))
    for episode in range(N_EPISODES):
        # initial observation
        observation = env.reset()
        rewards_sum = np.zeros(len(players))
        player1 = players[0]
        player2 = players[1]

        while True:
            # choose action based on observation
            actions = (player1.choose_action(observation),player2.choose_action(observation))

            # take action and get next observation and reward
            observation_, rewards, done = env.step(actions)
            rewards_sum += rewards

            player1.learn(observation, actions[0], rewards[0], observation_)
            player2.learn(observation, actions[1], rewards[1], observation_)

            # swap observation
            observation = observation_

            # break while loop when done
            if done:
                player1.learn(observation, actions[0], rewards[0], observation_,done)
                break

        # end of game
        print('Episode {} finished after {} steps.'.format(episode + 1, env.step_ctr))
        avg_rewards[:,episode] = rewards_sum * 1.0 / env.step_ctr
    return avg_rewards

def plot_results(avg_rewards, legend):
    plt.plot(avg_rewards[0,:],'b')
    plt.plot(avg_rewards[1,:],'g')
    plt.xlabel('Episode')
    plt.ylabel('Average reward per round')
    plt.legend(legend)
    plt.show()

if __name__ == "__main__":
    # Initialize env
    HISTORY_LENGTH = 5 # the NN will use the actions from this many past rounds to determine its action
    env = IPD(HISTORY_LENGTH)
    N_NODES = 16 #number of nodes in the intermediate layer of the NN
    # agent = DQN_Agent(env.n_actions, 2*HISTORY_LENGTH, N_NODES,
    #                   learning_rate=0.1,
    #                   reward_decay=0.9,
    #                   e_greedy=0.9,
    #                   replace_target_iter=20,
    #                   memory_size=2000,
    #                   )
    PG_agent = Policy_Gradient_Agent(env.n_actions, 2*HISTORY_LENGTH, N_NODES,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      )
    cooperateBot = CooperateBot()
    defectBot = DefectBot()
    tftBot = TitForTatBot(1)

    # Execute the algorithm
    N_EPISODES = 2000
    # avg_rewards = run_game(N_EPISODES,PG_agent,cooperateBot)
    # plot_results(avg_rewards,['DQN Agent','CooperateBot'])
    # PG_agent.reset()
    # avg_rewards = run_game(N_EPISODES,PG_agent,defectBot)
    # plot_results(avg_rewards,['DQN Agent','DefectBot'])
    # PG_agent.reset()
    avg_rewards = run_game(N_EPISODES,PG_agent,tftBot)
    plot_results(avg_rewards,['DQN Agent','TFTBot'])

