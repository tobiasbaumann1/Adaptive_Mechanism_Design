import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(filename='main.log',level=logging.DEBUG)
from Environments import Prisoners_Dilemma
from Agents import Actor_Critic_Agent, Critic_Variant, Simple_Agent
from Policing_Agent import Policing_Agent

HISTORY_LENGTH = 5 # the NN will use the actions from this many past rounds to determine its action
N_EPISODES = 2000
N_PLAYERS = 2
N_UNITS = 1 #number of nodes in the intermediate layer of the NN
#REWARD_STRENGTH = 6

def run_game(N_EPISODES, players, policing_agent = None, redistribution = True):
    env.reset_ep_ctr()
    avg_policing_rewards_per_round = []
    for episode in range(N_EPISODES):
        # initial observation
        s = env.reset()
        flag = isinstance(s, list)

        cum_policing_rs = [0]*len(players)
        while True:
            # choose action based on s
            if flag:
                actions = [player.choose_action(s[idx]) for idx, player in enumerate(players)]
            else:
                actions = [player.choose_action(s) for player in players]
            

            # take action and get next s and reward
            s_, rewards, done = env.step(actions)

            if policing_agent is not None:
                policing_rs = policing_agent.choose_action(s,actions)
                if redistribution:
                    sum_policing_r = sum(policing_rs)
                    mean_policing_r = sum_policing_r / N_PLAYERS
                    policing_rs = [r-mean_policing_r for r in policing_rs]
                rewards = [ sum(r) for r in zip(rewards,policing_rs)]
                cum_policing_rs = [sum(r) for r in zip(cum_policing_rs, policing_rs)]
                # Training policing agent
                policing_agent.learn(s,actions)
            logging.info('Actions:' + str(actions))
            logging.info('State after:' + str(s_))
            logging.info('Rewards: ' + str(rewards))
            logging.info('Done:' + str(done))

            for idx, player in enumerate(players):
                if flag:
                    player.learn(s[idx], actions[idx], rewards[idx], s_[idx], s, s_)
                else:
                    player.learn(s, actions[idx], rewards[idx], s_)

            # swap s
            s = s_

            # break while loop when done
            if done:
                for player in players:
                    player.learn_at_episode_end() 
                break

        avg_policing_rewards_per_round.append([r / env.step_ctr for r in cum_policing_rs])

        # status updates
        if (episode+1) % 100 == 0:
            print('Episode {} finished.'.format(episode + 1))
    return env.get_avg_rewards_per_round(), np.asarray(avg_policing_rewards_per_round)

def plot_results(avg_rewards_per_round, legend, title, ylabel = 'Reward', exp_factor = 1):
    plt.figure()
    for agent_idx in range(avg_rewards_per_round.shape[1]):
        avg = avg_rewards_per_round[0,agent_idx]
        avg_list = []
        for r in avg_rewards_per_round[:,agent_idx]:
            avg = exp_factor * r + (1-exp_factor) * avg
            avg_list.append(avg)
        first_idx = int(1 / exp_factor)
        plt.plot(range(first_idx,len(avg_list)),avg_list[first_idx:])
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.title(title)
    plt.savefig('./'+title)
    #plt.show()

def create_population(env,n_agents, use_simple_agents = False):    
    critic_variant = Critic_Variant.CENTRALIZED
    if use_simple_agents:
        l = [Simple_Agent(env, 
                      learning_rate=0.01,
                      gamma=0.9,
                      agent_idx = i,
                      critic_variant = critic_variant) for i in range(n_agents)]
    else:
        l = [Actor_Critic_Agent(env, 
                      learning_rate=0.01,
                      gamma=0.9,
                      n_units_actor = N_UNITS,
                      agent_idx = i,
                      critic_variant = critic_variant) for i in range(n_agents)]
    #Pass list of agents for centralized critic
    if critic_variant is Critic_Variant.CENTRALIZED:
        for agent in l:
            agent.pass_agent_list(l)
    return l

if __name__ == "__main__":
    # Initialize env and agents
    #env = Public_Goods_Game(HISTORY_LENGTH,N_PLAYERS, multiplier = 3, punishment_cost = 0.2, punishment_strength = 2)    
    # env = Prisoners_Dilemma(signal_possible = True, option_to_abstain = True)    
    # agents = create_population(env,2)
    
    # avg_rewards_per_round = 2 * run_game(N_EPISODES,agents)
    # plot_results(avg_rewards_per_round,[str(agent) for agent in agents],env.__str__(), exp_factor=0.1)
    # for agent in agents:
    #     agent.close()

    env = Prisoners_Dilemma()    
    agents = create_population(env,N_PLAYERS, use_simple_agents = True)
    policing_agent = Policing_Agent(env,agents)

    avg_rewards_per_round,avg_policing_rewards_per_round = run_game(N_EPISODES,agents,policing_agent, redistribution = False)
    plot_results(avg_rewards_per_round,[str(agent) for agent in agents],env.__str__(), exp_factor=0.1)
    plot_results(avg_policing_rewards_per_round,[str(agent) for agent in agents],env.__str__()+'_policing_rewards', exp_factor=0.1)
    actor_a_prob_each_round = np.transpose(np.array([agent.log for agent in agents]))
    plot_results(actor_a_prob_each_round,[str(agent) for agent in agents],env.__str__()+'_player_action_probabilities', ylabel = 'P(Cooperation)')
    policing_a_prob_each_round = np.array(policing_agent.get_log()[0])
    plot_results(policing_a_prob_each_round,['Agent plays D', 'Agent plays C'],env.__str__()+'_policing_action_probabilities', ylabel = 'P(a_p=1)')
