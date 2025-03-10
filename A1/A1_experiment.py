# from A1_env import CartPoleEnv
import A1_agent
from importlib import reload
reload(A1_agent)
from A1_agent import CartPoleAgent

import numpy as np
import gymnasium as gym



def experiment(n_timesteps = int(1e6), n_envs = 1000, gamma = 1, epsilon = 0.1, TN = False, ER = False):
    # env = CartPoleEnv(n_envs=n_envs)
    # states, info = CartPoleEnv.env.reset() 

    envs = gym.make_vec('CartPole-v1', num_envs= n_envs) # Next_step reset is default
    states, info = envs.reset() # Initial states
    DQN_agent = CartPoleAgent(envs, epsilon=0.1)

    steps = 0

    if ER:
        done = np.zeros(shape = n_envs, dtype = bool)

    while steps <= n_timesteps:
        actions = DQN_agent.select_action(states, policy = 'egreedy') # Decide action 
        states_next, rewards, terminations, truncations, infos = envs.step(actions)

        DQN_agent.update(states, rewards, states_next)
        
        states = states_next
        
        steps += np.sum(~done)
        
        if ER:
            done = np.logical_or(terminations, truncations)

    print(f'Took {steps} steps.')
    
    envs.close()

if __name__ == '__main__':
    experiment()