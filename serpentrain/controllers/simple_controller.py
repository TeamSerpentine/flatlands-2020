from functools import partial
from multiprocessing import Pool

from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv

from serpentrain.controllers.abstract_controller import AbstractController


class SimpleController(AbstractController):
    def __init__(self):
        super(SimpleController, self).__init__()
        self.env = None

        self.shortest_paths = []
        self.time_step = 0
        self.pool = Pool()

    def start_of_round(self, obs, env: RailEnv, **kwargs):
        """
        Gets called at the start of a round, with the obs, and a copy of the env
        """
        # We got the RailEnv
        self.env = env

        # Reset time
        self.time_step = 0

        # Planning ahead for all agents, disregarding collisions
        pool_shortest_path = partial(self.shortest_path, self.env.distance_map)
        self.shortest_paths = self.pool.map(pool_shortest_path, self.env.get_agent_handles())

    def act(self, observation):
        """
        Gets called each step, with a dict of observations for each agent
        """
        actions = {}
        for idx, agent in enumerate(self.env.agents):
            if agent.status not in (RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED):
                current_position, current_direction = agent.position, agent.direction
                pass
            else:
                actions[idx] = 0
        return actions

    def env_reaction(self, state, action, reward, next_state, done):
        """
        Gets called after act, with the SARSD update
        State[t], Action[t], Reward[t], State[t+1], Done[t+1]
        """
        self.time_step += 1

    def end_of_round(self):
        pass
