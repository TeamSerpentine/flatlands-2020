import sys
import time
import traceback
from random import randint, seed, SystemRandom

import numpy as np
from flatland.envs.rail_env import RailEnv
from torch.multiprocessing import Process

from serpentrain.reinforcement_learning.distributed.utils import create_env, create_controller, ModelWrapper
from serpentrain.reinforcement_learning.memory.trajectory import Trajectory


class EnvWorker(Process):
    """
    Env Workers continuously run and record episodes
        Sending state to backlog queue
        Waiting for the return in the return dict
        Take the returned action in the env
        Record trajectory per agent
    Put recorded trajectories on the buffer queue
    """

    def __init__(self, backlog_queue, return_dict, buffer_queue, unique_id, episode_length):
        super(EnvWorker, self).__init__()
        self.backlog_queue = backlog_queue
        self.return_dict = return_dict
        self.buffer_queue = buffer_queue
        self.id = unique_id
        self.shutdown = False

        self.env: RailEnv = None
        self.model_ref = None
        self.controller = None

        self.epsilon = 0.90
        self.epsilon_decay = 0.9

        self.episode_length = episode_length

    def run(self) -> None:
        # Set a random state independent for each thread
        rng = SystemRandom()
        seed_ = rng.randint(0, 2 ** 32 - 1)
        seed(seed_)
        np.random.seed(seed_)

        # Initialize Worker
        print(f"Worker {self.id} Reporting for duty! seed:{seed_}")
        try:
            self.env = create_env(seed=seed_)
            self.model_ref = ModelWrapper.wrap_model(self.backlog_queue, self.return_dict, self.id)
            self.controller = create_controller(self.model_ref)
            # Run episodes until shutdown
            while not self.shutdown:
                print(f"Worker {self.id} Starting a new episode!")
                start_time = time.time()
                trajectories, stat_string = self._run_episode()
                print(f"Worker {self.id} finished an episode in {time.time() - start_time} seconds")
                print(stat_string)
                start_time = time.time()
                self._add_to_buffer(trajectories)
                self._update_parameters()

        except Exception as e:
            print(f"Worker {self.id} crashed {e}")
            print(traceback.print_exception(*sys.exc_info()))
        print(f"Worker {self.id} Disengaged")

    def _run_episode(self, eval_=False) -> [Trajectory]:
        """
        Runs a single episode and collects the trajectories of each agent
        """
        if eval_:
            epsilon = 0
        else:
            epsilon = self.epsilon

        # Create and Start Environment
        self._prepare_env()
        obs, info = self.env.reset(regenerate_rail=True, regenerate_schedule=True,
                                   random_seed=True)
        score = 0
        trajectories = [Trajectory() for _ in self.env.get_agent_handles()]

        # Create and Start Controller
        self.controller.start_of_round(obs=obs, env=self.env)

        step = 0
        done = {}

        # Env has a max episode step, I am sure this is not intended but it might be useful
        max_steps = self.env._max_episode_steps if self.env._max_episode_steps is not None else self.episode_length

        for step in range(max_steps):
            action_dict, processed_obs = self.controller.act(observation=obs, info=info, epsilon=epsilon)
            next_obs, all_rewards, done, info = self.env.step(action_dict)

            # Save actions and rewards for each agent
            for agent_handle in self.env.get_agent_handles():
                if action_dict[agent_handle] is not None:
                    trajectories[agent_handle].add_row(
                        state=processed_obs[agent_handle],
                        action=action_dict[agent_handle],
                        reward=all_rewards[agent_handle],
                        done=done[agent_handle])

            score += sum(all_rewards)

            obs = next_obs.copy()
            if done['__all__'] or self.shutdown:
                break

        n_agents_done = sum(done.values()) - 1
        stat_string = f"total_score: {score} (avg: {score / self.env.get_num_agents()})\n" \
                      f"step: {step}. amount of agents done: {n_agents_done} out of {self.env.get_num_agents()}\n" \
                      f"Epsilon: {epsilon}"

        return trajectories, stat_string

    def _add_to_buffer(self, trajectories):
        for trajectory in trajectories:
            self.buffer_queue.put(trajectory)

    def _update_parameters(self):
        self.epsilon *= self.epsilon_decay

    def _prepare_env(self):
        self.env.number_of_agents = randint(1, 100)
        self.env.width = randint(10, 100)
        self.env.height = randint(10, 100)
