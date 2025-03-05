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
import time

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        T_ep = len(actions)
        gammas = np.power(np.full(T_ep,self.gamma),np.arange(0,T_ep)) #pre-calculate all possible values for gamma^i
        for t in range(T_ep):
            m = np.min((n,T_ep - t))
            if done and t+m == T_ep: #if s_{t+m} is terminal, e.g. was the last state even terminal and is t+m referring to the last state (index wise)?
                G = np.sum(gammas[:m-1] * rewards[t:t+m-1])
            else:
                G = np.sum(gammas[:m-1] * rewards[t:t+m-1]) + self.gamma**m * np.max(self.Q_sa[states[t+m]]) 
            
            self.Q_sa[states[t],actions[t]] += self.learning_rate * (G - self.Q_sa[states[t],actions[t]]) # Update rule


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=False, n=5, eval_interval=500, verbose = False):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
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
        
        pi.update(states, actions, rewards, done, n=n)

        if episode % eval_interval == 0:
            r_mean = pi.evaluate(eval_env, verbose = verbose)
            eval_returns.append(r_mean)
            eval_timesteps.append(episode)
            if verbose:
                print(f'EVALUATION at {episode = } yields mean return of {r_mean}')


        if plot:
           env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
        
    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 50001
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = False
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n, verbose= True)
    
    
if __name__ == '__main__':
    now = time.time()
    test()
    print(f'Code ran for {time.time() - now}')
