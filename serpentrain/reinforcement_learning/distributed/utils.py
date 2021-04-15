import time

from torch import tensor
from torch.types import Device

from serpentrain.models.model import Model


class ModelWrapper(Model):
    """
    Wraps a model for remote (another process) inference
    """

    def __init__(self, backlog_queue, return_dict, unique_id: int, device: Device):
        """
        A simple wrapper for a torch model that runs in a separate process
        """
        super().__init__(device)
        self.backlog_queue = backlog_queue
        self.return_dict = return_dict
        self.unique_id = unique_id

    def forward(self, x: tensor):
        x.share_memory_()
        job = (self.unique_id, x)
        self.backlog_queue.put(job)
        start_time = time.time()
        while self.unique_id not in self.return_dict and time.time() - start_time < 60:
            time.sleep(0.1)
        return self.return_dict.pop(self.unique_id)

    @classmethod
    def wrap_model(cls, backlog_queue, return_dict, unique_id):
        return ModelWrapper(backlog_queue, return_dict, unique_id, "cpu")


def create_model(state_dict=None, device="cuda:0"):
    """
    Helper function that creates the model everywhere
    This way it only needs to be defined here
    """
    from serpentrain.models.linear_model import LinearModel
    model = LinearModel(device=device, input_size=231, output_size=5, layer_sizes=[2048, 1024, 512, 128])
    if state_dict:
        model.load_state_dict(state_dict)
        model.eval()  # set dropout and batch normalization layers to evaluation mode
    return model


def create_replay_buffer():
    """
    Helper function that creates the replay buffer everywhere
    This way it only needs to be defined here
    """
    from serpentrain.reinforcement_learning.memory.simple_replay_buffer import SimpleReplayBuffer
    return SimpleReplayBuffer(buffer_size=10_000, batch_size=256)


def create_controller(model):
    """
    Helper function that creates the controller everywhere
    This way it only needs to be defined here

    When run in a env worker:
        The model should take care of sending and retrieving data to and from the inference server
    """
    from serpentrain.controllers.dqn_controller import DQNController
    return DQNController(model=model, action_size=5, obs_tree_depth=2)


def create_env(seed=None):
    """
    Helper function that creates an env everywhere
    This way it only needs to be defined here
    """
    from flatland.envs.rail_env import RailEnv
    from flatland.envs.observations import TreeObsForRailEnv
    from flatland.envs.rail_generators import complex_rail_generator
    from flatland.envs.schedule_generators import complex_schedule_generator
    # TODO make more configurable
    env = RailEnv(width=20,
                  height=20,
                  obs_builder_object=TreeObsForRailEnv(2),
                  rail_generator=complex_rail_generator(nr_start_goal=100,
                                                        nr_extra=2,
                                                        min_dist=8,
                                                        max_dist=99999,
                                                        seed=seed),
                  schedule_generator=complex_schedule_generator(seed=seed),
                  number_of_agents=3,
                  random_seed=seed
                  )
    return env
