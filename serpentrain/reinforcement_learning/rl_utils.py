from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv


def load_env(env_dict, obs_builder_object=GlobalObsForRailEnv()):
    """
    Loads an env
    """
    env = RailEnv(height=4, width=4, obs_builder_object=obs_builder_object)
    env.reset(regenerate_rail=False, regenerate_schedule=False)
    RailEnvPersister.set_full_state(env, env_dict)
    return env
