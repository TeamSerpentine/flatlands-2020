from functools import partial
from multiprocessing import Pool

from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_action_for_move

from serpentrain.controllers.abstract_controller import AbstractController


class WaitIfOccupiedAnywhereController(AbstractController):
    def __init__(self):
        super(WaitIfOccupiedAnywhereController, self).__init__()
        self.env = None

        self.shortest_paths = []
        self.time_step = 0
        self.pool = Pool()

        self.occupied = []
        self.height = 0
        self.width = 0
        self.agents_left = -1

        self.DEBUG = False

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

        # Occupied places
        self.height = env.height
        self.width = env.width
        self.occupied = [[False for w in range(env.width)] for h in range(env.height)]

        self.agents_left = len(env.active_agents)

    def act(self, observation):
        """
        Gets called each step, with a dict of observations for each agent
        """
        # Recalculate occupied places
        self.occupied = [[0 for j in range(self.height)] for i in range(self.width)]
        for idx, agent in enumerate(self.env.agents):
            agentPos = agent.position
            if agentPos is not None:
                self.occupied[agentPos[0]][agentPos[1]] = True

        # Check how many agents are left
        cur_agents_left = len(self.env.active_agents)
        if cur_agents_left < self.agents_left:
            self.agents_left = cur_agents_left
            print("#Agents left:", self.agents_left)

        # Find an action for each agent
        actions = {}
        blocked_agents = 0
        moving_agents = []
        for idx in self.env.active_agents:
            agent = self.env.agents[idx]
            if agent.status not in (RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED):
                current_position, current_direction = agent.position, agent.direction

                # If we are at the next spot, cut off the first spot so the current spot is at index 0 again
                while current_position is not None and len(self.shortest_paths[idx]) >= 2 \
                        and self.shortest_paths[idx][0] != (current_position, current_direction):
                    # Remove path we already have passed
                    self.shortest_paths[idx] = self.shortest_paths[idx][1:]

                # If we are at the destination already, move the last bit
                if len(self.shortest_paths[idx]) <= 1:
                    if self.DEBUG:
                        print("Moving already arrived train the last bit forward")
                    actions[idx] = RailEnvActions.MOVE_FORWARD
                    continue

                # Check the entire path for other agents
                pathBlocked = False
                for step in self.shortest_paths[idx][1:]:
                    position = step.position

                    # Do not check the position it is currently on itself (e.g. for roundabout turns)
                    if current_position is not None and \
                            current_position[0] == position[0] and \
                            current_position[1] == position[1]:
                        continue

                    # Check if the spot on the path is occupied
                    if self.occupied[position[0]][position[1]]:
                        pathBlocked = True
                        break

                if not pathBlocked:
                    # If no part of the path is blocked, move
                    actions[idx] = get_action_for_move(*self.shortest_paths[idx][0], *self.shortest_paths[idx][1],
                                                       self.env.rail)

                    moving_agents.append(idx)

                    # Mark the spot it will go to as occupied to prevent deadlock
                    if current_position is not None:
                        next_position = self.shortest_paths[idx][0].position
                        self.occupied[next_position[0]][next_position[1]] = True
                else:
                    # If part of the path is blocked, stop
                    actions[idx] = RailEnvActions.STOP_MOVING
                    blocked_agents += 1
            else:
                actions[idx] = RailEnvActions.MOVE_FORWARD

        if self.DEBUG:
            print("Moving agents:", moving_agents)

        # Detect deadlocks of all agents waiting on each other
        if blocked_agents == self.agents_left:
            if self.DEBUG:
                print("All agents are blocked, deadlock!")

            # As last resort, try and get out of deadlock by moving all forward
            # TODO Prevent deadlock
            for idx in self.env.active_agents:
                actions[idx] = RailEnvActions.MOVE_FORWARD

        return actions

    def env_reaction(self, state, action, reward, next_state, done):
        """
        Gets called after act, with the SARSD update
        State[t], Action[t], Reward[t], State[t+1], Done[t+1]
        """
        self.time_step += 1

    def end_of_round(self):
        pass
