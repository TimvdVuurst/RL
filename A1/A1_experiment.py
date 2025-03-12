# from A1_env import CartPoleEnv
import A1_agent
from importlib import reload
reload(A1_agent)
from A1_agent import CartPoleAgent

import numpy as np
import gymnasium as gym
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import Tensor


def experiment(n_timesteps = int(1e6), n_envs = 1000, n_eval_episodes = 50, gamma = 0.99,
                epsilon = 0.05, lr = 1e-3, TN = False, ER = False):
    envs = gym.make_vec('CartPole-v1', num_envs= n_envs, vectorization_mode='sync') # Next_step reset is default
    eval_envs = gym.make_vec('CartPole-v1', num_envs= n_eval_episodes) # Next_step reset is default

    #TODO: remove seed
    states, info = envs.reset(seed = 42) # Initial states with seed for reproducability 
    DQN_agent = CartPoleAgent(envs, gamma = gamma, epsilon = epsilon, lr = lr)


    # if ER:
    done = np.zeros(shape = n_envs, dtype = bool)

    steps = 0
    #TQDM bar
    pbar = tqdm(total = 100)
    prog = 0
    eval_rewards = []
    timesteps = []
    while steps <= n_timesteps:
        actions = DQN_agent.select_action(states, policy = 'egreedy') # Decide action 

        states_next, rewards, terminations, truncations, infos = envs.step(actions)

        DQN_agent.policy_net.train() # Set network to training mode
        DQN_agent.update(states, rewards, actions, states_next, done)

        states = states_next
        # print(type(states))
        
        done = np.logical_or(terminations, truncations)

        #Not sure if this is true
        if ER:
            steps += np.sum(~done)
        else:
            steps += n_envs 
        
        DQN_agent.policy_net.eval() # Set network to evaluation mode
        eval_rewards.append(DQN_agent.evaluate(eval_envs, eval_envs.observation_space.shape[0])) #Evaluate in every loop, e.g. after n_env steps
        timesteps.append(steps)

        #TQDM bar update
        prog_i = np.floor((steps/n_timesteps)*100).astype(int)
        pbar.update(prog_i - prog)
        prog = prog_i
    
    envs.close()
    pbar.close()

    plt.figure()
    plt.scatter(timesteps,eval_rewards, c = 'black')
    plt.xlabel('Number of steps')
    plt.ylabel('Mean evaluation reward over 50 episodes')
    plt.show()


if __name__ == '__main__':
    experiment()