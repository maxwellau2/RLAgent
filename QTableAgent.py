import numpy as np
import gym

class QTableAgent:
    def __init__(self, bins, state_bounds, action_size, learning_rate=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.bins = bins  # Number of bins per state variable
        self.state_bounds = state_bounds  # Clipping ranges for each state variable
        self.q_table = np.zeros(bins + [action_size])  # Multi-dimensional Q-table
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def discretize_state(self, state):
        """ Converts a continuous state into a discrete index. """
        discrete_state = []
        for i in range(len(state)):
            state_clipped = np.clip(state[i], self.state_bounds[i][0], self.state_bounds[i][1])  # Clip state
            bin_size = (self.state_bounds[i][1] - self.state_bounds[i][0]) / (self.bins[i] - 1)
            bin_index = int((state_clipped - self.state_bounds[i][0]) / bin_size)
            discrete_state.append(bin_index)
        return tuple(discrete_state)

    def act(self, state):
        discrete_state = self.discretize_state(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)  # Random action
        return np.argmax(self.q_table[discrete_state])  # Best action

    def remember(self, state, action, reward, next_state, done):
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        best_next_action = np.argmax(self.q_table[discrete_next_state])
        target = reward + (self.gamma * self.q_table[discrete_next_state + (best_next_action,)] * (not done))
        self.q_table[discrete_state + (action,)] += self.learning_rate * (target - self.q_table[discrete_state + (action,)])

    def decay_epsilon(self):
        """ Decay epsilon after each episode. """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        np.save(filepath, self.q_table)

    def load(self, filepath):
        self.q_table = np.load(filepath)
