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
        
        self.softmax = nn.Softmax(dim = 0) #change to dim = 1 for vectorized input
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CustomSquareLoss(nn.Module):
    def __init__(self):
        super(CustomSquareLoss, self).__init__()
    
    def forward(self, pred, target):
        return torch.sum(torch.square(pred - target))  # Custom loss

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
        self.loss_function = CustomSquareLoss()
    

            
    def select_action(self, s, policy='egreedy'):
        s = Tensor(s)
        num = self.env.observation_space.shape[0]
        policy = policy.lower()
        if policy == 'greedy':
            with torch.no_grad():
                a = torch_argmax(self.policy_net(s),dim = 1)

        elif policy == 'egreedy':
            if self.epsilon is None:
                raise KeyError("Provide an epsilon")
            
            if num > 1:
                a = np.zeros(num)
                random_mask = self.epsilon >= np.random.rand(num) 
                a[random_mask] = np.random.choice(self.n_actions, size = random_mask.sum())
                with torch.no_grad():
                    a[~random_mask] = torch.argmax(self.policy_net(s[~random_mask]), dim = 1)

            else:
                if self.epsilon >= np.random.rand():
                    a = np.random.choice(self.n_actions)
                else:
                    with torch.no_grad():
                        a = torch.argmax(self.policy_net(s))

        else:
            raise ValueError("Invalid policy given.")
        
        return np.array(a).astype(np.int32)
        
    def update(self):
        raise NotImplementedError('For each agent you need to implement its specific back-up method') 


    def evaluate(self,eval_env,n_eval_episodes=50, max_episode_length = 500, verbose = False):
        returns = np.zeros(shape=n_eval_episodes)  # list to store the reward per episode
        done = np.zeros(shape = n_eval_episodes, dtype = bool)

        s, _ = eval_env.reset() #initialize

        while (np.sum(done) < eval_env.observation_space.shape[0]): 
            a = self.select_action(s, 'greedy')
            s_prime, rewards, terminations, truncations, infos = eval_env.step(a)
            
            # returns[~done] += rewards[~done] # Only update rewards of non-terminated envs
            # s[~done] = s_prime[~done] # Only update non-terminated states
            
            returns += rewards
            s = s_prime # Only update non-terminated states
            
            done += np.logical_or(terminations, truncations) # Mask of terminated envs, checks termination and max_episode length inherently

        mean_return = np.mean(returns)
        return mean_return


class CartPoleAgent(BaseAgent):
    
    def update(self, states, rewards, actions, states_next, done, TN = False, ER = False):
        #TODO: check if this here is the best course of action or if it shouldnt be in experiemnt.py
        states = Tensor(states)
        rewards = Tensor(rewards)
        states_next = Tensor(states_next)

        action_index = torch.tensor([[int(a)] for a in actions]) 

        q_values = torch.flatten(torch.gather(self.policy_net(states),1,action_index)) # NN prediction over actions
        # Q-value prediction
        q_values_next = torch.flatten(torch.gather(self.policy_net(states_next),1,action_index))
        
        # Target rule by Minh et al 2013
        target = rewards.detach().clone() # It gets the reward regardless
        target[np.invert(done)] += self.gamma * q_values_next[np.invert(done)] # Additional info for non-terminal states i+1
        # target[done] = rewards[done]

        # Calculate loss
        loss = self.loss_function(q_values, target)

        #Backpropagate
        self.optimizer.zero_grad() # Reset gradients
        loss.backward() 
        self.optimizer.step()

    def update_single(self, states, rewards, states_next, done, TN = False, ER = False):
        #TODO: check if this here is the best course of action or if it shouldnt be in experiemnt.py
        states = Tensor(states)
        rewards = Tensor(rewards)
        states_next = Tensor(states_next)

        # Target rule by Minh et al 2013
        target = rewards.detach().clone() # It gets the reward regardless,

        for (s, r, s_next, d, t) in zip(states, rewards, states_next, done, target): #Loop over parallel environments

            Q = torch.max(self.policy_net(s))
            Q_next = torch.max(self.policy_net(s_next))

            if d: # for terminated next states
                t += self.gamma * Q_next
            
            loss = self.loss_function(Q, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



