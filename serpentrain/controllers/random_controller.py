import random

import numpy as np
from flatland.envs.rail_env import RailEnv

from serpentrain.controllers.abstract_controller import AbstractController


class RandomController(AbstractController):
    def __init__(self, action_size):
        self.time_step = 0
        self.num_round = 0
        self.env = None
        self.action_size = action_size

    def start_of_round(self, obs, env: RailEnv, **kwargs):
        self.time_step = 0
        self.num_round += 1
        self.env = env

    def end_of_round(self):
        pass

    def act(self, observation):
        """
        Gets called each step, with a dict of observations for each agent.
        Computes dict of actions of all agents
        """
        actions = {}
        for idx, agent in enumerate(self.env.agents):
            actions[idx] = random.choice(np.arange(self.action_size))
        return actions

    def env_reaction(self, state, action, reward, next_state, done):
        """
        Gets called after act, with the SARSD update
        State[t], Action[t], Reward[t], State[t+1], Done[t+1]
        """
        self.time_step += 1
