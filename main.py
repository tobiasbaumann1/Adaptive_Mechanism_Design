import matplotlib.pyplot as plt
import numpy as np
import os
import logging
logging.basicConfig(filename='main.log',level=logging.DEBUG,filemode='w')
from Environments import Prisoners_Dilemma
from Agents import Actor_Critic_Agent, Critic_Variant, Simple_Agent
from Planning_Agent import Planning_Agent

N_EPISODES = 2000
N_PLAYERS = 2
N_UNITS = 10 #number of nodes in the intermediate layer of the NN
MAX_REWARD_STRENGTH = 3

def run_game(N_EPISODES, players, planning_agent = None, with_redistribution = True):
    env.reset_ep_ctr()
    avg_planning_rewards_per_round = []
    for episode in range(N_EPISODES):
        # initial observation
        s = env.reset()
        flag = isinstance(s, list)

        cum_planning_rs = [0]*len(players)
        while True:
            # choose action based on s
            if flag:
                actions = [player.choose_action(s[idx]) for idx, player in enumerate(players)]
            else:
                actions = [player.choose_action(s) for player in players]
            

            # take action and get next s and reward
            s_, rewards, done = env.step(actions)

            if planning_agent is not None:
                planning_rs = planning_agent.choose_action(s,actions)
                if with_redistribution:
                    sum_planning_r = sum(planning_rs)
                    mean_planning_r = sum_planning_r / N_PLAYERS
                    planning_rs = [r-mean_planning_r for r in planning_rs]
                rewards = [ sum(r) for r in zip(rewards,planning_rs)]
                cum_planning_rs = [sum(r) for r in zip(cum_planning_rs, planning_rs)]
                # Training planning agent
                planning_agent.learn(s,actions)
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
        avg_planning_rewards_per_round.append([r / env.step_ctr for r in cum_planning_rs])

        # status updates
        if (episode+1) % 100 == 0:
            print('Episode {} finished.'.format(episode + 1))
    return env.get_avg_rewards_per_round(), np.asarray(avg_planning_rewards_per_round)

def plot_results(avg_rewards_per_round, legend, path, title, ylabel = 'Reward', exp_factor = 1):
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
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path+'/' + title)
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

def run_game_and_plot_results(env,agents, 
    with_redistribution = False, max_reward_strength = None, cost_param = 0):
    planning_agent = Planning_Agent(env,agents,max_reward_strength = max_reward_strength, 
        cost_param = cost_param, with_redistribution = with_redistribution)
    avg_rewards_per_round,avg_planning_rewards_per_round = run_game(N_EPISODES,agents,planning_agent, 
        with_redistribution = with_redistribution)
    path = './Results/different_seed/with' + ('' if with_redistribution else 'out') + '_redistribution' 
    path += '/' + 'max_reward_strength_' + (str(max_reward_strength) if max_reward_strength is not None else 'inf')
    path += '/' + 'cost_parameter_' + str(cost_param)

    plot_results(avg_rewards_per_round,[str(agent) for agent in agents],path,env.__str__(), exp_factor=0.1)
    plot_results(avg_planning_rewards_per_round,[str(agent) for agent in agents],path,env.__str__()+'_planning_rewards', exp_factor=0.1)
    actor_a_prob_each_round = np.transpose(np.array([agent.log for agent in agents]))
    plot_results(actor_a_prob_each_round,[str(agent) for agent in agents],path,env.__str__()+'_player_action_probabilities', ylabel = 'P(Cooperation)')
    planning_a_prob_each_round = np.array(planning_agent.get_log())
    plot_results(planning_a_prob_each_round,['(D,D)', '(C,D)', '(D,C)', '(C,C)'],path,env.__str__()+'_planning_action', ylabel = 'a_p')
    
if __name__ == "__main__":

    env = Prisoners_Dilemma()    
    agents = create_population(env,N_PLAYERS, use_simple_agents = True)
    run_game_and_plot_results(env,agents,with_redistribution=False, max_reward_strength = 3, cost_param = 0)    