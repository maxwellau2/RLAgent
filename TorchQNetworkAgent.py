import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from agent import RLAgent

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class TorchQNetworkAgent(RLAgent):
    def __init__(self, state_size, action_size, device="cpu"):
        super().__init__(state_size, action_size)
        self.device = device
        self.memory = deque(maxlen=2000)
        self.model = DQNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        # Sample a random batch from memory
        batch = random.sample(self.memory, batch_size)
        
        # Convert batch into separate lists
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to PyTorch tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)  # Shape: (batch_size, state_size)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device).view(-1, 1)  # Shape: (batch_size, 1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device).view(-1, 1)  # Shape: (batch_size, 1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)  # Shape: (batch_size, state_size)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device).view(-1, 1)  # Shape: (batch_size, 1)

        # Compute Q-values for current states
        q_values = self.model(states)  # Shape: (batch_size, action_size)

        # Gather Q-values for the taken actions
        current_q_values = q_values.gather(1, actions)  # Shape: (batch_size, 1)

        # Compute max Q-value for next states (ignoring actions)
        with torch.no_grad():
            max_next_q_values = self.model(next_states).max(dim=1, keepdim=True)[0]  # Shape: (batch_size, 1)

        # Compute the target Q-value using Bellman equation
        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
