#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        self.reset_Delta()
    
    def reset_Delta(self):
        self.Delta = 0

    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        a = np.argmax(self.Q_sa[s])
        return a
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        x = np.copy(self.Q_sa[s,a])
        self.Q_sa[s,a] = np.sum(p_sas * (r_sas + self.gamma * np.max(self.Q_sa,axis=1))) #Eq 1
        self.Delta = np.max((self.Delta,np.abs(x-self.Q_sa[s,a]))) #Update max error
    
     
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

    i = 0
    while QIagent.Delta > threshold or i == 0: #Stopping (and starting) condition
        QIagent.reset_Delta() #set Delta to 0  
        #Loop over all states and actions
        for s in range(QIagent.n_states):
            for a in range(QIagent.n_actions):
                QIagent.update(s,a,env.p_sas[s,a,:],env.r_sas[s,a,:]) # Slicing??
        
        # Plot current Q-value estimates & print max error
        print("Q-value iteration, iteration {}, max error {}".format(i,QIagent.Delta))
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.001)
        i += 1 
 
    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)

    # view optimal policy
    done = False
    s = env.reset()
    rs = []
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        rs.append(r)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        s = s_next

    # TODO: Compute mean reward per timestep under the optimal policy
    print("Mean reward per timestep under optimal policy: {}".format(np.mean(rs)))
    print("Sum of rewards under optimal policy: {}".format(np.sum(rs)))
    
if __name__ == '__main__':
    experiment()
