import gymnasium as gym
import numpy as np

class CartPoleEnv:

    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.env = gym.make_vec('CartPole-v1', num_envs = self.n_envs)
    
    def __repr__(self):
        print(f"Vectorized CartPole environment with N = {self.n_envs} environments in Sync mode.")
    
    def step(self, a):
        '''
        a is array of actions or single action integer
        '''
        if isinstance(a,int):
            a = np.full(self.n_envs, a) # If given a single value for action, take that action for all environments
        
        observations, rewards, terminations, truncations, infos = self.env.step(a)

        return observations, rewards, terminations, truncations, infos


if __name__ == '__main__':
    test_env = CartPoleEnv(5)
    print(test_env.env.observation_space.shape[0])
    print(test_env.env.single_action_space.n)
