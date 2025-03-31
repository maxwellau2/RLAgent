from collections import namedtuple, deque
import math
import torch
import random as rnd
import torch.nn as nn
import gymnasium as gym

# main source:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# chcek device, i have a gtx3050
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transition for replay memory
Transition = namedtuple("Transition", ('state', 'action', 'next_state', 'reward'))

# helper class to store transitions
class ReplayMemory:
    def __init__(self, capacity: int = 10000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int = 128):
        return rnd.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
# our funciton approximator to the Q table
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)
    

# defining the DQNAgent

class DQNAgent:
    def __init__(self,
                 env: gym.Env,
                 batch_size: int = 128,
                 gamma: float = 0.99,
                 eps_start: float = 0.9,
                 eps_end: float = 0.05,
                 eps_decay: float = 1000,
                 tau: float = 0.005,
                 lr: float = 1e-4,
                 ):
        self.env: gym.Env = env
        self.batch_size: int = batch_size
        self.gamma: float = gamma
        # we will use a episilon threshold custom formula to decay
        self.eps_start: float = eps_start
        self.eps_end: float = eps_end
        self.eps_decay: float = eps_decay

        # for soft update of the target network
        self.tau: float = tau
        self.lr: float = lr
        self.steps_done: int = 0

        # some setup stuff like getting actions and observations
        self.n_actions: int = self.env.action_space.n # possible actions should be 2
        self.n_observations = self.env.observation_space.shape[0] # 4 observable factors

        # initialising the "Q table" aka policy networks
        # we use the policy net to make the decisions
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        # we update the target net from policy net
        # this is because we wish the target net to be stable, 
        # as directly retraining the policy net can result in "catastrophic forgetting"
        # source: https://stackoverflow.com/questions/54237327/why-is-a-target-network-required
        # Conceptually it's like saying, "I have an idea of how to play this well, 
        # I'm going to try it out for a bit until I find something better" 
        # as opposed to saying "I'm going to retrain myself how to play this 
        # entire game after every move". By giving your network more time to 
        # consider many actions that have taken place recently instead of 
        # updating all the time, it hopefully finds a more robust model before you 
        # start using it to make actions.
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # target network is not trained directly
        # it is instead, used only for inference; not updated via backpropagation
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.replay_memory = ReplayMemory(10000)

    def select_action(self, state: torch.Tensor, random=True) -> torch.Tensor:
        """Selects an action using an epsilon-greedy policy."""
        if not random:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        sample = rnd.random()
        # this is our exponential epsilon "decay strategy"
        # f(x)=0.05+(0.9−0.05)*e(-x/1000)
        # i want to start at 0.9, and end at 0.05
        # to visualise the decay strategy, you can plot it on desmos by pasting the following...
        # \ f\left(x\right)=0.05\ +\ \left(0.9\ -\ 0.05\right)\ \cdot\left(e\right)^{\left(\frac{-x}{1000}\right)}
        # i took it straight from the sample code in pytorch docs
        # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        # the idea behind it is we do not want the decay to be too fast to explore more
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        # print(eps_threshold, sample)
        if sample > eps_threshold: 
            # look up q table, the .no_grad() context manager disables 
            # back_propagation, making it faster for general inference
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else: # random walk
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)
        
    def optimize_model(self):
        """Runs one step of optimization on the policy network."""
        if len(self.replay_memory) < self.batch_size:
            return
        transitions = self.replay_memory.sample(self.batch_size)
        # transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        # taken straight from pytorch docs
        batch = Transition(*zip(*transitions))

        # we need to remove all the terminal states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # converting into a big matrix for pytorch to multiply,
        # will help train faster and more efficiently
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # gather the Q-values corresponding to the actions taken
        # see https://stackoverflow.com/questions/50999977/what-does-gather-do-in-pytorch-in-layman-terms
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V value (next state value)
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # compute Huber loss
        # for some reason, this works better than the MSE loss
        # according to some scources, huber loss is better for noisy environments
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # avoid gradient explosion using clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # soft update of the target network's weights
        self.soft_update_target_network()

    def soft_update_target_network(self):
        """Soft update the target network weights."""
        # soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        # refer to pytorch sample code
        # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            # slowly adjust the weight of the target network by a factor of 1-tau
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        # reload the weights from the dictionary
        self.target_net.load_state_dict(target_net_state_dict)

    def train(self, num_episodes: int=600):
        """Trains the agent for a given number of episodes."""
        episode_durations = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            total_reward = 0
            episode_finish = False
            t=0
            while not episode_finish:
                action: torch.Tensor = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # store in replay memory
                self.replay_memory.push(state, action, next_state, reward)

                # move to the next state
                state = next_state
                # only optimize every 4 ticks
                if t%4 == 0:
                    # Perform one step of the optimization (on the policy network)
                    self.optimize_model()

                total_reward += reward.item()
                t += 1
                if done:
                    episode_durations.append(t)
                    episode_finish = True
                    break
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        print("Training complete")

    def play(self, num_episodes=100, render=False) -> list:
        """
            Plays the game for a given number of episodes and returns the history of rewards as a list[int]
        """
        reward_history = []
        for episode in range(num_episodes):
            state = self.env.reset()
            if isinstance(state, tuple):  # handle older Gym versions
                state = state[0]
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            total_reward = 0
            episode_finish = False
            t = 0
            while not episode_finish:
                action: torch.Tensor = self.select_action(state, random=False)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # render for visualization
                if render:
                    self.env.render()

                # move to the next state
                state = next_state
                total_reward += reward.item()
                t += 1
                if done:
                    episode_finish = True
                    break
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")
            reward_history.append(total_reward)
        print("Playing complete")
        self.env.close()
        return reward_history

    def save_model(self, filepath="dqn_model.pth"):
        """Saves the trained model."""
        torch.save(self.policy_net.state_dict(), filepath)

    def load_model(self, filepath="dqn_model.pth"):
        """Loads a trained model."""
        self.policy_net.load_state_dict(torch.load(filepath))
        self.policy_net.to(device)
        self.policy_net.eval()


# ---- Running the training ----
if __name__ == "__main__":
    try:
        env = gym.make("CartPole-v1")
        agent = DQNAgent(env)

        agent.train(num_episodes=10_000)
    except KeyboardInterrupt:
        print("Exiting")
    finally:
        agent.save_model("DQN_MODEL2_pth")
        env.close()