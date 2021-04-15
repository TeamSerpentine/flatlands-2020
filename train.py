import cv2
import numpy as np
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.rendertools import RenderTool

from serpentrain.controllers.random_controller import RandomController


def log_video(_images, epoch):
    height, width, depth = _images[0].shape
    print(len(_images), height, width, depth)
    out = cv2.VideoWriter(f'video_{epoch}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    [out.write(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)) for image in _images]
    out.release()


def log_video(_images, epoch):
    height, width, depth = _images[0].shape
    print(len(_images), height, width, depth)
    out = cv2.VideoWriter(f'video_{epoch}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    [out.write(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)) for image in _images]
    out.release()


np.random.seed(1)

# Use the complex_rail_generator to generate feasible network configurations with corresponding tasks
# Training on simple small tasks is the best way to get familiar with the environment

obs_builder = GlobalObsForRailEnv()
env = RailEnv(width=20, height=20,
              rail_generator=complex_rail_generator(nr_start_goal=100, nr_extra=2, min_dist=8, max_dist=99999),
              schedule_generator=complex_schedule_generator(), obs_builder_object=obs_builder,
              number_of_agents=3)
env.reset()

env_renderer = RenderTool(env)

# Import your own Agent or use RLlib to train agents on Flatland
# As an example we use a random agent here
agent_kwargs = {"state_size": 0, "action_size": 5}
controller = RandomController(5)

n_trials = 5

# Empty dictionary for all agent action
action_dict = dict()
print("Starting Training...")

for trial in range(1, n_trials + 1):

    # Reset environment and get initial observations for all agents
    obs, info = env.reset()
    controller.start_of_round(obs, env)

    for idx in range(env.get_num_agents()):
        tmp_agent = env.agents[idx]
        tmp_agent.speed_data["speed"] = 1 / (idx + 1)
    env_renderer.reset()
    # Here you can also further enhance the provided observation by means of normalization
    # See training navigation example in the baseline repository
    images = []
    score = 0
    # Run episode
    for step in range(1000):
        # Chose an action for each agent in the environment
        action_dict = controller.act(observation=obs)

        # Environment step which returns the observations for all agents, their corresponding
        # reward and whether their are done
        next_obs, all_rewards, done, _ = env.step(action_dict)
        env_renderer.render_env(show=False, show_observations=False, show_predictions=False)
        images.append(env_renderer.get_image())
        score += sum(all_rewards)

        # Update replay buffer and train agent
        controller.env_reaction(obs, action_dict, all_rewards, next_obs, done)

        obs = next_obs.copy()
        if done['__all__']:
            break
    controller.end_of_round()
    print('Episode Nr. {}\t Score = {}'.format(trial, score))

    log_video(images, trial)
