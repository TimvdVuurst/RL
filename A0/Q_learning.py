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
from matplotlib import pyplot as plt
from Helper import argmax

class QLearningAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,done):
        # if done:
        #     return
        a_prime = argmax(self.Q_sa[s_next]) # Optimal choice at the next state
        G = r + self.gamma * self.Q_sa[s_next,a_prime]
        self.Q_sa[s,a] += self.learning_rate * (G - self.Q_sa[s,a])

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500, verbose = False):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    
    s = env.reset() # Initial state
    for episode in range(n_timesteps): 
        a = agent.select_action(s,policy,epsilon,temp) # Sample action
        s_prime, r, done = env.step(a) # Sample environment from that action
        agent.update(s,a,r,s_prime,done) # Update the Qvalues
        if done: # If s' reached the goal, reset
            s = env.reset()
        else:
            s = s_prime        

        if episode % eval_interval == 0:
            r_mean = agent.evaluate(eval_env)
            eval_returns.append(r_mean)
            eval_timesteps.append(episode)
            if verbose:
                print(f'EVALUATION at {episode = } yields mean return of {r_mean}')

        if plot:
           env.render(Q_sa=agent.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution


    return np.array(eval_returns), np.array(eval_timesteps)   

def test():
    
    n_timesteps = 10000
    eval_interval=100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # or 'greedy' or 'softmax'/'boltzmann'
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = False

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    print(eval_returns,eval_timesteps)

    plt.figure()
    plt.plot(eval_timesteps,eval_returns, c = 'black')
    plt.xlabel('Evaluation episode')
    plt.ylabel('Mean return per evaluation')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    test()
