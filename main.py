import numpy as np
import gym
from QTableAgent import QTableAgent

# Create Gym environment
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")

# Define discretization parameters
state_bounds = [
    (-4.8, 4.8),    # Cart Position
    (-10.0, 10.0),    # Cart Velocity (Clipped)
    (-0.418, 0.418),  # Pole Angle
    (-10.0, 10.0)     # Pole Angular Velocity (Clipped)
]
bins = [200, 100, 90, 100]  # Number of bins for each state variable

# Initialize Q-learning agent
state_size = len(bins)
action_size = env.action_space.n  # 2 actions (left, right)
agent = QTableAgent(bins, state_bounds, action_size, learning_rate=0.05, gamma=0.8, epsilon_decay=0.999, epsilon_min=0.01)

num_episodes = 20000

for episode in range(num_episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0
    render_this_episode = (episode + 1) % 100 == 0

    while not done:
        action = agent.act(state)  # Choose action
        next_state, reward, done, _, _ = env.step(action)  # Take action
        agent.remember(state, action, reward, next_state, done)  # Store experience
        state = next_state
        total_reward += reward
        # if render_this_episode:
            # env.render()
        # if reward == 0:
        #     print(f"we made it! {episode}")

    agent.decay_epsilon()
    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
    if total_reward>200:
        print("Episode Completed Successfully on Episode: ", episode)

env.close()
agent.save("q_table_cartpole.npy")
