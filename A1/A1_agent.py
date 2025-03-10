import gymnasium as gym
import numpy as np
import torch
from torch import nn, relu, optim, Tensor
from torch import argmax as torch_argmax


def argmax(x):
    ''' Own variant of np.argmax with random tie breaking. By Thomas Moerland (2023) '''
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except:
        return np.argmax(x)

class DQN(nn.Module):
    # Highly inspired by: https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae 
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class BaseAgent:
    # Highly inspired by Thomas Moerland's implementation (RL course 2023)

    def __init__(self, env, lr = 0.1, gamma = 1, epsilon = None):
        # self.n_states = n_states
        self.n_actions = env.single_action_space.n

        self.env = env # Vectorized environment
        self.input_dim = env.observation_space.shape[1] # Input is state, e.g. 4 for cartpole
         # Initialize the NN as defined in the DQN class, output is prob dist over possible actions, so output dim = 2
        self.policy_net = DQN(self.input_dim, self.n_actions)
      
        # Hyperparameters
        self.learning_rate = lr
        self.gamma = gamma
        self.epsilon = epsilon

        #Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.learning_rate)
        # self.loss_function = nn.MSELoss()
            
    def select_action(self, s, policy='egreedy'):
        s = Tensor(s)
        policy = policy.lower()
        if policy == 'greedy':
            a = torch_argmax(self.policy_net(s),dim = 1)

        elif policy == 'egreedy':
            if self.epsilon is None:
                raise KeyError("Provide an epsilon")
            
            elif self.epsilon >= np.random.rand(): # Random possibility for random motion
                a = np.random.choice(self.n_actions)
            else:
                a = torch_argmax(self.policy_net(s),dim = 1)  # Otherwise greedy choice
           
        else:
            raise ValueError("Invalid policy given.")
        
        return np.array(a)
        
    def update(self):
        raise NotImplementedError('For each agent you need to implement its specific back-up method') 


    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100, verbose = False):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = self.select_action(s, 'greedy')
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return


class CartPoleAgent(BaseAgent):
    
    def update(self, states, rewards, states_next, TN = False, ER = False):
        #TODO: check if this here is the best course of action or if it shouldnt be in experiemnt.py
        states = Tensor(states)
        states_next = Tensor(states_next)

        q_values = self.policy_net(states) # NN prediction over actions

        # Q-value prediction
        q_values_next = self.policy_net(states_next) # NN prediction of following states
        target = rewards + self.gamma * np.max(q_values_next, axis = 1) # (moving) target value

        # Calculate loss
        loss = nn.MSE()(q_values, target)

        #Backpropagate
        self.optimizer.zero_grad() # Reset gradients
        loss.backwards() 
        self.optimizer.step() 

    # def backpropagate(self,losses):



