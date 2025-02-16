import gym
import numpy as np
import random
import torch  # or tensorflow.keras
from collections import deque

from TorchQNetworkAgent import TorchQNetworkAgent

# Initialize environment
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
print(state_size)
action_size = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize DQN Agent
agent = TorchQNetworkAgent(state_size, action_size, device=device)  # Or TFQNetworkAgent

num_episodes = 500
batch_size = 32

for episode in range(num_episodes):
    state = env.reset()[0]
    state = np.reshape(state, [1, state_size])  # Ensure correct shape
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)  # Select action
        next_state, reward, done, _, _ = env.step(action)  # Take action
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)  # Store experience
        state = next_state
        total_reward += reward

        agent.learn(batch_size)  # Train model

    agent.decay_epsilon()  # Reduce exploration

    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

# Save trained model
agent.save("dqn_model.pth")  # For PyTorch, or "dqn_model.h5" for TensorFlow

env.close()
