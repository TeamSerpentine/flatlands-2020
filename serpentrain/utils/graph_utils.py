from collections import deque
from typing import Tuple, Set

import numpy as np


class Graph:
    """
    A graph representation of the rail network
    """
    direction_to_r_location = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1)
    }
    r_location_to_direction = {
        (-1, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (0, -1): 3
    }

    def __init__(self, transition_map: [[int]]):
        # self.grid_map: {Tuple[int, int]: Union[Graph.Node, Graph.edge]} = {}
        self.grid_map = np.empty((len(transition_map), len(transition_map[0])), dtype=Graph.Element)
        self.nodes: [Graph.Node] = []
        self.edges: [Graph.Edge] = []
        self.searched_locations: Set[Tuple[int, int]] = set()
        self.transition_map = transition_map
        self.floyd_warshal = None

        for x, row in enumerate(transition_map):
            for y, cell in enumerate(row):
                loc = (x, y,)
                if loc not in self.searched_locations:
                    if 1 in cell:
                        self.trace_transition(loc, cell)
                    else:
                        self.grid_map[x, y] = 0

        print(f"Amount of nodes {len(self.nodes)}")
        print(f"Amount of edges {len(self.edges)}")

    def trace_transition(self, location: Tuple[int, int], transitions, edge=None):
        """
        Traces a transition and creates nodes or edges, which recursively trace connections
        """
        transitions = np.reshape(np.asarray([int(bit) for bit in transitions], dtype=np.bool), (4, 4))
        count = np.count_nonzero(transitions)
        self.searched_locations.add(location)
        if count == 1:
            # Dead end
            node = Graph.Node(location, transitions, idx=len(self.nodes))
            self.grid_map[location] = node
            self.nodes.append(node)
        elif count == 2:
            # Straight

            if edge is None:
                connected_locations = deque()
                edge = Graph.Edge(location, transitions)
                self.grid_map[location] = edge
                self.edges.append(edge)
                connected_locations.append(edge.start_loc)
                connected_locations.append(edge.end_loc)
            else:
                self.grid_map[location] = edge
                return edge.add_location(location, transitions)

            while connected_locations:
                location = connected_locations.pop()
                if location not in self.searched_locations:
                    connected_transitions = self.transition_map[location[0]][location[1]]
                    loc = self.trace_transition(location, connected_transitions, edge)
                    if loc is not None:
                        connected_locations.append(loc)
            edge.finalise(self.grid_map)
        else:
            # Special Crossing
            node = Graph.Node(location, transitions, idx=len(self.nodes))
            self.grid_map[location] = node
            self.nodes.append(node)

    def loc_is_free_at_t(self, loc, t):
        """
        Checks if a loc is free at a given moment
        """
        element: Graph.Element = self.grid_map[loc[0], loc[1]]
        return element.is_free_at_t(t)

    def get_floyd_warshall(self):
        """
        Returns the floyd warshall distance matrix, makes it if it is not yet made
        """
        if self.floyd_warshal is None:
            self._make_floyd_warshall()
        return self.floyd_warshal

    def _make_floyd_warshall(self):
        """
        Creates the floyd-warshall distance matrix
        """
        n_nodes = len(self.nodes)
        self.floyd_warshal = np.full((n_nodes, n_nodes), 999999, dtype=np.int)

        for edge in self.edges:
            self.floyd_warshal[edge.start_node.idx, edge.end_node.idx] = edge.length
            self.floyd_warshal[edge.start_node.idx, edge.start_node.idx] = 0
            self.floyd_warshal[edge.end_node.idx, edge.end_node.idx] = 0

        for k in range(n_nodes):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    self.floyd_warshal[i, j] = min(self.floyd_warshal[i, j],
                                                   self.floyd_warshal[i, k] + self.floyd_warshal[k, j])

    @staticmethod
    def directions_after_transition(transition, direction) -> [int]:
        """
        :param transition: Transition matrix
        :param direction: incoming direction
        """
        x = np.array(np.nonzero(transition[direction, :])).ravel()
        return x

    @staticmethod
    def directions_before_transition(transition, direction) -> [int]:
        """
        :param transition: Transition matrix
        :param direction: incoming direction
        """
        x = np.array(np.nonzero(transition[:, direction])).ravel()
        return x

    @staticmethod
    def add_direction_to_location(location, direction) -> Tuple[int, int]:
        """
        :param location: x, y
        :param direction: int encoding for NESW
        """
        r_location = Graph.direction_to_r_location[direction]
        return location[0] + r_location[0], location[1] + r_location[1]

    @staticmethod
    def r_location(a, b):
        return a[0] - b[0], a[1] - b[1]

    class Element:
        def is_free_at_t(self, t):
            raise NotImplemented("Child should override this")

    class Node(Element):
        def __init__(self, location, transitions, idx):
            # Connections is a 2 d array with 1 if it is possible to go from incoming direction to outgoing location.
            # Coded as NESW. e.g. if connections[0, 0], a train facing north can go north and will face north after exit
            # and if connections[1,0], a train facing north can go
            # e.g.
            #    N  E  S  W - facing
            # N[[1 ,0 ,0 ,0],   [[1, 1, 1, 1],
            # E [0, 0, 0, 0],    [1, 1, 1, 1],
            # S [0, 0, 1, 0],    [1, 1, 1, 1],
            # W [0, 0, 0, 0]]    [1, 1, 1, 1]]
            #    N-S-Straight       Crossing
            # Occupations keeps track of the amount of trains wanting to pass at a certain time step with the same
            # encoding as Connections but with an int instead of a bool
            self.connections = np.zeros((4, 4), dtype=bool)
            self.occupations = {}
            self.idx = idx
            self.neighbours = None

            self.length = 1
            # X, Y location of this node
            self.location = np.array(location)

            self.transitions = transitions
            self.edges = []

        def add_edge(self, edge, incoming_direction):
            """
            Add a edge to this node
            """
            self.edges.append(edge)
            return len(self.edges) - 1

        def get_neighbours(self):
            if self.neighbours is None:
                self.neighbours = []
                for edge in self.edges:
                    if edge.start_node is not self:
                        self.neighbours.append(edge.end_node)
                    else:
                        self.neighbours.append(edge.start_node)

            return self.neighbours

    class Edge(Element):
        def __init__(self, location: Tuple[int, int], transitions):
            directions = np.nonzero(transitions)
            self.start_direction = directions[1][1]
            self.end_direction = directions[1][0]
            self.start_loc: Tuple[int, int] = Graph.add_direction_to_location(location, self.start_direction)
            self.start_node: Graph.Node = None
            self.end_loc: Tuple[int, int] = Graph.add_direction_to_location(location, self.end_direction)
            self.end_node: Graph.Node = None

            # The set of x, y coordinates this node occupies
            self.path = deque([location])
            self.locations: Set[Tuple[int, int]] = {location}

            # Node0_direction the direction the train is facing when entering node0
            self.start_idx = None
            self.end_idx = None

            # key: time_step, value: -1 occupied from end to start, 0 empty, 1 occupied from start to end
            self.occupation = {}
            self.length = None

            self.finalized = False

        def __len__(self):
            return self.length

        def is_free_at_t(self, t):
            return self.occupation.get(t, 0) == 0

        def add_location(self, location, transition) -> Tuple[int, int]:
            """
            Adds a location to this edge
            """
            if not self.finalized:
                self.locations.add(location)
                if self.start_loc == location:
                    self.path.appendleft(location)
                    direction = np.nonzero(transition[self.start_direction])[0][0]
                    self.start_loc = Graph.add_direction_to_location(location, direction)
                    self.start_direction = direction
                    return self.start_loc
                elif self.end_loc == location:
                    self.path.append(location)
                    direction = np.nonzero(transition[self.end_direction])[0][0]
                    self.end_loc = Graph.add_direction_to_location(location, direction)
                    self.end_direction = direction
                    return self.end_loc
                else:
                    RuntimeError("Cant add in the middle of edge")
            else:
                raise RuntimeError("Can't Add location to finalized graph")

        def finalise(self, grid_map):
            self.start_node: Graph.Node = grid_map[self.start_loc[0], self.start_loc[1]]
            self.end_node: Graph.Node = grid_map[self.end_loc[0], self.end_loc[1]]

            # Node0_direction the direction the train is facing when entering node0
            self.start_idx = self.start_node.add_edge(self, self.start_direction)
            self.end_idx = self.end_node.add_edge(self, self.end_direction)

            self.length = len(self.path)
            self.finalized = True
