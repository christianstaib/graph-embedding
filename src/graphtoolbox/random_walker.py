import random
import networkx as nx
from typing import List


class RandomWalker:
    def __init__(self) -> None:
        pass

    def one_random_walk(
        self,
        graph: nx.Graph,
        limit_length: bool = False,
        max_length: int = None
    ) -> List[int]:
        """Performs a random walk on a graph

        Args:
            graph (nx.Graph): Graph on which the random walk shall be done.
            limit_length (bool, optional): Indicates if the length of the walk is limited. Wether the length is limited or not, the random walk stops once a vertex is visited a second time. Defaults to False.
            max_length (int, optional): Max length of random walk. Defaults to None.

        Returns:
            List[int]: List of vertex IDs visited by the random walk.
        """

        walk = []
        walk.append(random.choice(range(graph.number_of_nodes())))

        while (not limit_length) or (len(walk) < max_length):
            neighborhood = list(graph.neighbors(walk[-1]))
            neighborhood = list(set(neighborhood) - set(walk))
            if neighborhood:
                walk.append(random.choice(neighborhood))
            else:
                break

        return walk

    def random_walks(
        self,
        graph: nx.Graph,
        num_walks: int,
        limit_length: bool = False,
        max_length: int = None
    ) -> List[List[int]]:
        """Performs num_walks many random walks on a given graph.

        Args:
            graph (nx.Graph): Graph on which the random walks shall be done.
            num_walks (int): Number of walks to be performed on the graph.
            limit_length (bool, optional): Indicates if the length of the walk is limited. Wether the length is limited or not, the random walk stops once a vertex is visited a second time. Defaults to False.
            max_length (int, optional): Max length of random walk. Defaults to None.

        Returns:
            List[List[int]]: List of List of vertex IDs visited by the random walks.
        """

        walks = []

        for _ in range(num_walks):
            walk = self.one_random_walk(
                graph,
                limit_length,
                max_length)
            walks.append(walk)

        return walks
