import time
from typing import Callable

from flatland.utils.rendertools import RenderTool

from serpentrain.controllers.abstract_controller import AbstractController
from serpentrain.reinforcement_learning.memory.trajectory import Trajectory
from serpentrain.reinforcement_learning.rl_utils import load_env


def run_episode(kwargs) -> [Trajectory]:
    """
    Runs a single episode and collects the trajectories of each agent
    """
    total_controller_time = 0
    env_dict: Callable = kwargs.get("env_dict")
    obs_builder = kwargs.get("obs_builder")
    controller_creator: Callable = kwargs.get("controller_creator")
    episode_id: int = kwargs.get("episode_id")
    max_episode_length: int = kwargs.get("max_episode_length", 1000)
    render: bool = kwargs.get("render", False)
    # Create and Start Environment
    _env = load_env(env_dict, obs_builder_object=obs_builder)
    obs, info = _env.reset(regenerate_rail=False, regenerate_schedule=True, )
    score = 0
    _trajectories = [Trajectory() for _ in _env.get_agent_handles()]

    # Create and Start Controller
    controller: AbstractController = controller_creator()
    start = time.time()
    controller.start_of_round(obs=obs, env=_env)
    total_controller_time += time.time() - start

    if render:
        env_renderer = RenderTool(_env)
        env_renderer.reset()

    for step in range(max_episode_length):
        start = time.time()
        action_dict, processed_obs = controller.act(observation=obs)
        total_controller_time += time.time() - start
        next_obs, all_rewards, done, info = _env.step(action_dict)

        if render:
            env_renderer.render_env(show=True, show_observations=True, show_predictions=False)

        # Save actions and rewards for each agent
        [_trajectories[agent_handle].add_row(
            state=processed_obs[agent_handle],
            action=action_dict[agent_handle],
            reward=all_rewards[agent_handle],
            done=done[agent_handle])
            for agent_handle in _env.get_agent_handles()]

        score += sum(all_rewards)

        obs = next_obs.copy()
        if done['__all__']:
            break

    if render:
        env_renderer.close_window()
    # print(f"\nController took a total time of: {total_controller_time} seconds", flush=True)
    return _trajectories
