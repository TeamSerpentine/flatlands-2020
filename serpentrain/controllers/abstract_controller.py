from abc import ABC, abstractmethod

from flatland.envs.rail_env_shortest_paths import get_shortest_paths

from serpentrain.models.model import Model
from serpentrain.reinforcement_learning.memory.replay_buffer import ReplayBuffer
import warnings


class AbstractController(ABC):
    @abstractmethod
    def act(self, observation):
        """
        Gets Called every step
        """

    @abstractmethod
    def env_reaction(self, state, action, reward, next_state, done):
        """
        Gives info on the last made action
        A tuple of State[t] (obs), Action[t], Reward[t], State[t+1] (next_obs), Done[t+1]
        each a dict with key: agent_handle, and there respective values
        """

    @abstractmethod
    def start_of_round(self, obs, env, **kwargs):
        """
        Gets called at the start of a new round with the possibility of some base info
        """

    @abstractmethod
    def end_of_round(self):
        """
        Gets called at the end of the round
        """

    @staticmethod
    def train(model: Model, replay_buffer: ReplayBuffer):
        """
        Train the model with samples from the replay buffer
        """
        warnings.warn("Training has not been implemented for this controller")

    @staticmethod
    def shortest_path(distance_map, handle):
        """
        Calculates the naive shortest path of an agent
        """
        return get_shortest_paths(distance_map, agent_handle=handle)[handle]

    @staticmethod
    def first_conflict(path1, path2, offset1=0, offset2=0):
        """
        Calculates time and position of first conflict between two paths.
        TODO: I assume speed=1.0

        Parameters
        ----------
        path1 : list
        path2 : list
        offset1 : int
            Time of first step of path1 (default = 0)
        offset2 : int
            Time of first step of path2 (default = 0)

        Returns
        -------
        conflict : bool
        time : int
            Time of first conflict
        position : tuple
            Position (x,y) of first conflict
        """
        offset = max(offset1, offset2)
        for t in range(offset, min(len(path1) + offset1, len(path2) + offset2)):
            t1 = t - offset1
            t2 = t - offset2

            if path1[t1].position == path2[t2].position:
                return True, t, path1[t1].position

            # https://discourse.aicrowd.com/t/train-close-following
            # 'Tailgaters' never hit the train in front of them, regardless of order.
            # Therefore, the only remaining case is trains that 'swap' position.
            if t > offset and path1[t1].position == path2[t2 - 1].position and path1[t1 - 1].position == path2[
                t2].position:
                # the returned position is not 100% accurate, since the trains collide 'in the middle'
                return True, t, path1[t1].position

        return False
