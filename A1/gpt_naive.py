import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NUM_ENVS = 1000
GAMMA = 0.99
LEARNING_RATE = 1e-3
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

# Initialize network
policy_net = DQN(num_states, num_actions).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(0, num_actions, size=(NUM_ENVS,))
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            return policy_net(state_tensor).argmax(dim=1).cpu().numpy()

# Training loop
state, _ = env.reset()
epsilon = EPSILON_START
step = 0
episode_rewards = []

for episode in range(1000):
    state, _ = env.reset()
    episode_reward = 0
    
    for t in range(500):
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = np.logical_or(terminated, truncated)
        
        # Compute Q-value target
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
        done_tensor = torch.tensor(done, dtype=torch.float32, device=device)
        
        q_values = policy_net(state_tensor).gather(1, torch.tensor(action, dtype=torch.long, device=device).unsqueeze(1)).squeeze(1)
        next_q_values = policy_net(next_state_tensor).max(1)[0].detach()
        target_q_values = reward_tensor + (1 - done_tensor) * GAMMA * next_q_values
        
        loss = F.mse_loss(q_values, target_q_values.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
        episode_reward += reward.mean()
        
        step += 1
        epsilon = max(EPSILON_END, EPSILON_START - step / EPSILON_DECAY)
        
        if done.all():
            break
    
    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

env.close()
