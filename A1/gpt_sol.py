import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from torch.distributions import Categorical
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NUM_ENVS = 1000
GAMMA = 0.99
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
TARGET_UPDATE_FREQ = 100
MEMORY_SIZE = 10000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 10000

# Create vectorized environment
env = gym.make_vec('CartPole-v1', num_envs=NUM_ENVS)
num_actions = env.single_action_space.n
num_states = env.single_observation_space.shape[0]

# Q-Network class
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Initialize networks
policy_net = DQN(num_states, num_actions).to(device)
target_net = DQN(num_states, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        for i in range(NUM_ENVS):
            self.buffer.append((state[i], action[i], reward[i], next_state[i], done[i]))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.tensor(state, dtype=torch.float32, device=device),
                torch.tensor(action, dtype=torch.long, device=device),
                torch.tensor(reward, dtype=torch.float32, device=device),
                torch.tensor(next_state, dtype=torch.float32, device=device),
                torch.tensor(done, dtype=torch.float32, device=device))
    
    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer(MEMORY_SIZE)

# Epsilon-greedy action selection
def select_action(state, epsilon):
    if random.random() < epsilon:
        return np.random.randint(0, num_actions, size=(NUM_ENVS,))
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            return policy_net(state_tensor).argmax(dim=1).cpu().numpy()

# Training loop
state, _ = env.reset()
epsilon = EPSILON_START
step = 0

while len(replay_buffer) < MIN_REPLAY_SIZE:
    action = select_action(state, epsilon)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = np.logical_or(terminated, truncated)
    replay_buffer.push(state, action, reward, next_state, done)
    state = next_state

episode_rewards = []
for episode in range(1000):
    state, _ = env.reset()
    episode_reward = 0
    
    for t in range(500):
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = np.logical_or(terminated, truncated)
        
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward.mean()
        
        if len(replay_buffer) > BATCH_SIZE:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.sample(BATCH_SIZE)
            
            q_values = policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
            next_q_values = target_net(batch_next_state).max(1)[0].detach()
            target_q_values = batch_reward + (1 - batch_done) * GAMMA * next_q_values
            
            loss = F.mse_loss(q_values, target_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        step += 1
        epsilon = max(EPSILON_END, EPSILON_START - step / EPSILON_DECAY)
        
        if done.all():
            break
    
    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

env.close()
