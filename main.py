import matplotlib.pyplot as plt
from Environments import Prisoners_Dilemma
from Agents import Actor_Critic_Agent, Critic_Variant, Reputation_Bot

HISTORY_LENGTH = 5 # the NN will use the actions from this many past rounds to determine its action
N_EPISODES = 500
N_PLAYERS = 4
N_UNITS = 64 #number of nodes in the intermediate layer of the NN

def run_game(N_EPISODES, players):
    env.reset_ep_ctr()
    for episode in range(N_EPISODES):
        # initial observation
        s = env.reset()
        flag = isinstance(s, list)

        while True:
            # choose action based on s
            if flag:
                actions = [player.choose_action(s[idx]) for idx, player in enumerate(players)]
            else:
                actions = [player.choose_action(s) for player in players]
            

            # take action and get next s and reward
            s_, rewards, done = env.step(actions)
            # print('Actions:',actions)
            # print('State after:',s_)
            # print('Rewards:',rewards)
            # print('Done:',done)

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

        # status updates
        if (episode+1) % 10 == 0:
            print('Episode {} finished.'.format(episode + 1))
    return env.get_avg_rewards_per_round()

def plot_results(avg_rewards_per_round, legend, label, exp_factor = 1):
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
    plt.ylabel('Reward')
    plt.legend(legend)
    plt.title(label)
    plt.savefig('./'+label)
    #plt.show()

def create_population(env,n_actor_critic_agents,n_reputation_bots = 0):    
    critic_variant = Critic_Variant.CENTRALIZED
    l = [Actor_Critic_Agent(env, 
                  learning_rate=0.01,
                  gamma=0.9,
                  agent_idx = i,
                  critic_variant = critic_variant) for i in range(n_actor_critic_agents)]
    l.extend([Reputation_Bot(env,i+n_actor_critic_agents) for i in range(n_reputation_bots)])
    #Pass list of agents for centralized critic
    if critic_variant is Critic_Variant.CENTRALIZED:
        for agent in l:
            agent.pass_agent_list(l)
    return l

if __name__ == "__main__":
    # Initialize env and agents
    #env = Public_Goods_Game(HISTORY_LENGTH,N_PLAYERS, multiplier = 3, punishment_cost = 0.2, punishment_strength = 2)    
    env = Prisoners_Dilemma(signal_possible = True, option_to_abstain = True)    
    agents = create_population(env,2)
    
    avg_rewards_per_round = 2 * run_game(N_EPISODES,agents)
    plot_results(avg_rewards_per_round,[str(agent) for agent in agents],env.__str__(), exp_factor=0.1)
    for agent in agents:
        agent.close()

    env = Prisoners_Dilemma(signal_possible = True, option_to_abstain = False)    
    agents = create_population(env,2)
    
    avg_rewards_per_round = 2 * run_game(N_EPISODES,agents)
    plot_results(avg_rewards_per_round,[str(agent) for agent in agents],env.__str__(), exp_factor=0.1)