import random

import numpy as np
import torch
import torch.nn.functional as F
from flatland.envs.rail_env import RailEnv
from torch import optim
from torch.optim import Optimizer

from fl_utils.observation_utils import normalize_observation
from serpentrain.controllers.abstract_controller import AbstractController
from serpentrain.models.linear_model import LinearModel
from serpentrain.models.model import Model
from serpentrain.reinforcement_learning.memory.replay_buffer import ReplayBuffer


class DQNController(AbstractController):
    def __init__(self, model: LinearModel, action_size=5, obs_tree_depth=2, obs_radius=10):
        self.model = model
        self.action_size = action_size
        self.obs_tree_depth = obs_tree_depth
        self.obs_radius = obs_radius

        self.time_step = 0
        self.num_round = 0
        self.env = None
        self.actions = dict()

    def start_of_round(self, obs, env: RailEnv, **kwargs):
        self.time_step = 0
        self.num_round += 1
        self.env = env
        for idx, agent in enumerate(env.agents):
            self.actions[idx] = 0

    def end_of_round(self):
        pass

    def act(self, observation, info=None, epsilon=0):
        """
        Gets called each step, with a dict of observations for each agent.
        Computes dict of actions of all agents
        """
        self.time_step += 1
        processed_obs = {}
        for idx, agent in enumerate(self.env.agents):
            if observation[idx] is not None and info is None or info['action_required'][idx]:
                # Prepare observation
                state = observation[idx]
                state = normalize_observation(state, self.obs_tree_depth, observation_radius=self.obs_radius)
                processed_obs[idx] = state
                state = torch.from_numpy(state).float().unsqueeze(0)
                state = state.to(self.model.device)

                # Predict values of actions
                with torch.no_grad():
                    action_values = self.model(state)

                # Epsilon-greedy action selection
                if random.random() > epsilon:
                    action = np.argmax(action_values.cpu().data.numpy())
                else:
                    action = random.choice(np.arange(self.action_size))

                self.actions[idx] = action
            else:
                processed_obs[idx] = None
                self.actions[idx] = None

        return self.actions, processed_obs

    def env_reaction(self, state, action, reward, next_state, done):
        """
        Gets called after act, with the SARSD update
        State[t], Action[t], Reward[t], State[t+1], Done[t+1]
        """
        self.time_step += 1

    @staticmethod
    def train(model: Model, replay_buffer: ReplayBuffer, gamma: float = 0.99, optimizer: Optimizer = None):
        """
        Train the model with samples from the replay buffer
        """
        # Set the model to training mode
        model.train()

        # If no optimizer is given, create one
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=0.5e-4)

        # Get observations to train on
        states, actions, rewards, next_states, dones = replay_buffer.sample()

        # Store the observations in the device (cpu/gpu) we are running the model on
        states = states.to(model.device)
        actions = actions.to(model.device)
        rewards = rewards.to(model.device)
        next_states = next_states.to(model.device)
        dones = dones.to(model.device)

        # Get expected Q values of the actions that were taken
        # Basically, predict the value of taking the action that was taken in the state it was in
        q_expected = model(states).gather(1, actions)

        # Get the highest Q value of any action of the next state
        # Basically, for each possible action, predict its value of taking such action in the state it ended up in
        # and take the highest one of those to get a prediction of the highest value you could reach from the next
        # state
        # detach removes/strips away the information about gradients and such
        # max(1) takes for each output layer (dimension 1), the maximum value and the index at which this value is
        # [0] then selects that we look at the values, not the indices
        # unsqueeze transforms these values to tensors (on the last dimension so each output layer gets 1 tensor)
        # (think of it like 5.unsqueeze() => tensor(value = 5))
        q_targets_next = model(next_states).detach().max(1)[0].unsqueeze(-1)

        # Compute target Q values of the actions that were taken
        # Basically, based on the reward we got for taking an action and how good we consider the state we ended up in,
        # compute what value we give to taking the action in the current state, so what we want that prediction to be
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Compute loss with MSE
        # Basically, this computes how similar the target (the value we want) is to the expected
        # (the value we currently predict). The lower this is, the better we consider the model to be.
        loss = F.mse_loss(q_expected, q_targets)

        # Train: minimise the loss
        # Basically, the model will update its parameters such that the loss will be lower for the examples we show it
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Compute gradients from
        optimizer.step()  # Update parameters of model to minimize loss (train)

        # Set the model back to evaluation mode
        model.eval()

        return loss
