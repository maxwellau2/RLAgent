import numpy as np
from abc import ABC, abstractmethod



# this class serves as the foundational steps to train an RL Agent
# Implmentations of Q Tables or DQN are derived from this class
class RLAgent(ABC):
    
    def __init__(self, state_size: int,
                 action_size: int, 
                 learning_rate: float = 0.01, 
                 gamma: float = 0.99, 
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Abstract base class for RL Agents.

        Parameters:
        - state_size (int): Size of the state space
        - action_size (int): Number of possible actions
        - learning_rate (float): Learning rate for the optimizer
        - gamma (float): Discount factor for future rewards
        - epsilon (float): Initial exploration rate
        - epsilon_decay (float): Decay rate for epsilon
        - epsilon_min (float): Minimum value for epsilon
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    @abstractmethod
    def act(self, state: np.ndarray) -> int:
        """
        Abstract method for selecting an action given a state.
        
        Parameters:
        - state (np.ndarray): State to act on
        
        Returns:
        - action (int): Action to take
        """
        pass

    @abstractmethod
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Abstract method for storing a transition in the agent's memory.
        
        Parameters:
        - state (np.ndarray): State at the beginning of the transition
        - action (int): Action taken in the transition
        - reward (float): Reward received in the transition
        - next_state (np.ndarray): State at the end of the transition
        - done (bool): Whether the transition is terminal
        """
        pass

    @abstractmethod
    def learn(self):
        """
        Trains the agent using the stored experiences.
        """
        pass

    def decay_epsilon(self):
        """ Decays epsilon after each episode to reduce exploration over time. """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @abstractmethod
    def save(self, filepath):
        """
        Saves the model or Q-table to a file.

        Parameters:
        - filepath (str): Path to save the model.
        """
        pass

    @abstractmethod
    def load(self, filepath):
        """
        Loads the model or Q-table from a file.

        Parameters:
        - filepath (str): Path to load the model from.
        """
        pass
    