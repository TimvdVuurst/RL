#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import time
import os

from Q_learning import q_learning
from SARSA import sarsa
from Nstep import n_step_Q
from MonteCarlo import monte_carlo
from Helper import LearningCurvePlot, smooth
from tqdm import tqdm

def average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, gamma, policy='egreedy', 
                    epsilon=None, temp=None, smoothing_window=None, plot=False, n=5, eval_interval=500, verbose = False):

    returns_over_repetitions = []   
    now = time.time()
    
    for rep in tqdm(range(n_repetitions)): # Loop over repetitions
        if backup == 'q':
            returns, timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval, verbose)
        elif backup == 'sarsa':
            returns, timesteps = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
        elif backup == 'nstep':
            returns, timesteps = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n, eval_interval)
        elif backup == 'mc':
            returns, timesteps = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, eval_interval)

        returns_over_repetitions.append(returns)
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    learning_curve = np.mean(np.array(returns_over_repetitions),axis=0) # average over repetitions  
    if smoothing_window is not None: 
        learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve, timesteps  

def experiment():
    ####### Settings
    # Experiment      
    n_repetitions = 20
    smoothing_window = 9 # Must be an odd number. Use 'None' to switch smoothing off!
    plot = False # Plotting is very slow, switch it off when we run repetitions
    
    # MDP    
    n_timesteps = 50001 # Set one extra timestep to ensure evaluation at start and end
    eval_interval = 1000
    max_episode_length = 100
    gamma = 1.0
    
    # Parameters we will vary in the experiments, set them to some initial values: 
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.05
    temp = 1.0
    # Back-up & update
    backup = 'q' # 'q' or 'sarsa' or 'mc' or 'nstep'
    learning_rate = 0.1
    n = 5 # only used when backup = 'nstep'
        
    # Nice labels for plotting
    backup_labels = {'q': 'Q-learning',
                  'sarsa': 'SARSA',
                  'mc': 'Monte Carlo',
                  'nstep': 'n-step Q-learning'}
    
    ####### Experiments
    
    #### Assignment 1: Dynamic Programming
    # Execute this assignment in DynamicProgramming.py
    optimal_episode_return = 84 # set the optimal return per episode you found in the DP assignment here
    
    #### Assignment 2: Effect of exploration
    print(20*'='+' Assignment 2 ' + 20*'=')
    cd = os.getcwd()
    policy = 'egreedy'
    epsilons = [0.03,0.1,0.3]
    learning_rate = 0.1
    backup = 'q'
    Plot = LearningCurvePlot(title = 'Exploration: $\epsilon$-greedy versus softmax exploration') 
    Plot.set_ylim(-100, 100) 
    for epsilon in epsilons:
        print(f'Now running {epsilon = }')    
        filename = os.path.join(cd,r'A0\qlearning',f'lc_epsilon_{epsilon}.npy')
        filename_timesteps = os.path.join(cd,r'A0\qlearning',f'lc_epsilon_{epsilon}_time.npy')
        if os.path.isfile(filename):
            learning_curve = np.load(filename)
            timesteps = np.load(filename_timesteps)
        else:
            learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval, verbose = False)
            np.save(filename,learning_curve)
            np.save(filename_timesteps,timesteps)

        Plot.add_curve(timesteps,learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))    

    policy = 'softmax'
    temps = [0.01,0.1,1.0]
    print('E-GREEDY COMPLETED')
    for temp in temps:
        print(f'Now running {temp = }')
        filename = os.path.join(cd,r'A0\qlearning',f'lc_softmax_{temp}.npy')
        filename_timesteps = os.path.join(cd,r'A0\qlearning',f'lc_softmax_{temp}_time.npy')

        if os.path.isfile(filename):
            learning_curve = np.load(filename)
            timesteps = np.load(filename_timesteps)
        else:
            learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval, verbose = False)
            np.save(filename,learning_curve)
            np.save(filename_timesteps,timesteps)

        Plot.add_curve(timesteps,learning_curve,label=r'softmax, $ \tau $ = {}'.format(temp))

    Plot.add_hline(optimal_episode_return, label="DP optimum")
    Plot.save(os.path.join(cd,'A0','exploration.png'))

    # return
        
    ###### Assignment 3: Q-learning versus SARSA
    print(20*'='+' Assignment 3 ' + 20*'=')
    policy = 'egreedy' 
    epsilon = 0.1 # set epsilon back to original value 
    learning_rates = [0.03,0.1,0.3]
    backups = ['q','sarsa']
    Plot = LearningCurvePlot(title = 'Back-up: on-policy versus off-policy')    
    Plot.set_ylim(-100, 100) 
    for backup in backups:
        print(F'\nNOW RUNNING {backup} learning')
        for learning_rate in learning_rates:
            print(f'{learning_rate = }')
            filename = os.path.join(cd,'A0','q_vs_sarsa',f'lc_{backup}_{learning_rate}.npy')
            filename_timesteps = os.path.join(cd,'A0','q_vs_sarsa',f'lc_{backup}_{learning_rate}_time.npy')

            if os.path.isfile(filename):
                learning_curve = np.load(filename)
                timesteps = np.load(filename_timesteps)
            else:
                learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                    gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval, verbose = False)
                np.save(filename,learning_curve)
                np.save(filename_timesteps,timesteps)

            Plot.add_curve(timesteps,learning_curve,label=r'{}, $\alpha$ = {} '.format(backup_labels[backup],learning_rate))
    Plot.add_hline(optimal_episode_return, label="DP optimum")
    Plot.save(os.path.join(cd,'A0','on_off_policy.png'))
    

    # ##### Assignment 4: Back-up depth
    print(20*'='+' Assignment 4 ' + 20*'=')
    policy = 'egreedy'
    epsilon = 0.05 # set epsilon back to original value
    learning_rate = 0.1
    backup = 'nstep'
    ns = [1,3,10]
    Plot = LearningCurvePlot(title = 'Back-up: depth')   
    Plot.set_ylim(-100, 100) 
    for n in ns:
        print(f'Now working on Nstep with {n = }')
        filename = os.path.join(cd,'A0','nstep_mc',f'lc_nstep_{n}.npy')
        filename_timesteps = os.path.join(cd,'A0','nstep_mc',f'lc_nstep_{n}_time.npy')
        if os.path.isfile(filename):
            learning_curve = np.load(filename)
            timesteps = np.load(filename_timesteps)
        else:
            learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                    gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval)
            np.save(filename,learning_curve)
            np.save(filename_timesteps,timesteps)
        
        Plot.add_curve(timesteps,learning_curve,label=r'{}-step Q-learning'.format(n))

    print('Now working on MC')
    backup = 'mc'   
    filename = os.path.join(cd,'A0','nstep_mc',f'lc_mc.npy')
    filename_timesteps = os.path.join(cd,'A0','nstep_mc',f'lc_mc_time.npy')
    if os.path.isfile(filename):
        learning_curve = np.load(filename)
        timesteps = np.load(filename_timesteps)
    else:
        learning_curve, timesteps = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                gamma, policy, epsilon, temp, smoothing_window, plot, n, eval_interval)
        np.save(filename, learning_curve)
        np.save(filename_timesteps, timesteps)

    Plot.add_curve(timesteps,learning_curve,label='Monte Carlo')        
    Plot.add_hline(optimal_episode_return, label="DP optimum")
    Plot.save(os.path.join(cd,'A0','depth.png'))

if __name__ == '__main__':
    experiment()
