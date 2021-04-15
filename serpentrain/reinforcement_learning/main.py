from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from operator import itemgetter
from time import time

import torch
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from torch import distributed
from torch.multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm

from reinforcement_learning.random_policy import RandomPolicy
from serpentrain.controllers.dqn_controller import DQNController
from serpentrain.models.linear_model import LinearModel
from serpentrain.reinforcement_learning.episode_runner import run_episode
from serpentrain.reinforcement_learning.memory.replay_buffer import ReplayBuffer
from serpentrain.reinforcement_learning.memory.simple_replay_buffer import SimpleReplayBuffer

policy_save_file = f"checkpoints/{datetime.now().strftime('%y%m%d%H%M%S')}/random"


def env_creator():
    """
    Creates an env and returns it
    """
    return RailEnv(
        width=20,
        height=30,
        rail_generator=complex_rail_generator(nr_start_goal=100,
                                              nr_extra=2,
                                              min_dist=8,
                                              max_dist=99999,
                                              seed=False),
        schedule_generator=complex_schedule_generator(seed=False),
        obs_builder_object=GlobalObsForRailEnv(),
        number_of_agents=3,
        random_seed=True)


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('-e', '--episodes', default=10, type=int,
                        help='number of episodes per epoch (default: %(default)s)', metavar='E')
    parser.add_argument('-l', '--episode-length', '--length', default=500, type=int,
                        help='max number of steps per episode (default: %(default)s)', metavar='L')
    parser.add_argument('-n', '--epochs', default=10, type=int,
                        help='number of training iterations (default: %(default)s)', metavar='N')
    parser.add_argument('-r', '--render', action='store_true', help='render the environment')
    parser.add_argument('-w', '--workers', default=(cpu_count() - 1), type=int,
                        help='number of parallel workers (default: cores minus one)', metavar='W')
    parser.add_argument("-m", "--multiprocess", action="store_true",
                        help="Use multiprocessing to play multiple episodes at the same time")
    parser.add_argument("--gpu", action="store_true", help="Use gpu for model training and inference")
    episodes, episode_length, epochs, render, n_workers, multiprocess, gpu = \
        itemgetter('episodes', 'episode_length', 'epochs', 'render', 'workers',
                   "multiprocess", "gpu")(vars(parser.parse_args()))

    policy = RandomPolicy()

    start_time = time()

    if gpu:
        print(f"Cuda available: {torch.cuda.is_available()}")
        torch.cuda.set_device(0)
        print(f"Cuda initialised: {torch.cuda.is_initialized()}")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = LinearModel(device=device, input_size=231, layer_sizes=[5], output_size=1)

    replay_buffer: ReplayBuffer = SimpleReplayBuffer(buffer_size=10000, batch_size=16)

    base_env = env_creator()
    base_env.reset()
    base_env_dict = RailEnvPersister.get_full_state(env=base_env)

    controller_arguments = {
        "model": model,
        "action_size": 5,
    }

    controller_creator = partial(DQNController, **controller_arguments)
    master_controller = controller_creator()

    if multiprocess:
        try:
            print(f"Distributed available: {distributed.is_available()}")
            set_start_method("spawn")
            master_controller.model.share_memory()
        except Exception as e:
            print(f"Could not share memory: {e}")
    for epoch in range(epochs):
        pool_start = time()
        # Create pickable episode kwargs
        episodes_kwargs = [
            {
                "env_dict": base_env_dict,
                "obs_builder": TreeObsForRailEnv(2),
                "controller_creator": controller_creator,
                "max_episode_length": 1000,
                "render": False,
                "episode_id": epoch * episodes + episode
            }
            for episode in range(episodes)]
        epoch_trajectories = []
        if multiprocess:
            with Pool(n_workers) as workers:
                for episode_trajectories in tqdm(workers.imap_unordered(func=run_episode, iterable=episodes_kwargs),
                                                 total=episodes):
                    epoch_trajectories += episode_trajectories
        else:
            for episode_kwargs in tqdm(episodes_kwargs, total=episodes):
                epoch_trajectories += run_episode(episode_kwargs)

        # Add trajectories to replay buffer
        for trajectory in epoch_trajectories:
            for state, action, reward, next_state, done in trajectory.as_rows():
                replay_buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

        DQNController.train(model=model, replay_buffer=replay_buffer)

        print(f"pool took: {time() - pool_start}, total took {time() - start_time}")
        print(f"one episode took: {(time() - pool_start) / episodes}")
