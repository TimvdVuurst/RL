#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep '''
        t = len(states) - 1
        Gs = np.full(t + 1,0) # len T_ep + 1 filled with zeros for ease
        # print(f'{t=}, {len(states)=}, {len(actions)=},{len(rewards)=}')
        for i in reversed(range(0,t)):
            Gs[i] = rewards[i] + self.gamma*Gs[i+1]
            self.Q_sa[states[i],actions[i]] += self.learning_rate * (Gs[i] - self.Q_sa[states[i],actions[i]])


def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500, verbose = False):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    for episode in range(n_timesteps):
        states, actions, rewards = [], [], []
        states.append(env.reset())
        for t in range(max_episode_length):
            a = pi.select_action(states[t], policy, epsilon, temp)
            s_prime, r, done = env.step(a) # Sample environment 
            states.append(s_prime); actions.append(a); rewards.append(r)
            if done:
                break
        
        pi.update(states, actions, rewards)

        if episode % eval_interval == 0:
            r_mean = pi.evaluate(eval_env, verbose = verbose)
            eval_returns.append(r_mean)
            eval_timesteps.append(episode)
            if verbose:
                print(f'EVALUATION at {episode = } yields mean return of {r_mean}')


        if plot:
           env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) ## Plot the Q-value estimates during Monte Carlo RL execution
    
    return np.array(eval_returns), np.array(eval_timesteps)  

                 
    
def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = False

    monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, verbose=True)
    
            
if __name__ == '__main__':
    test()
