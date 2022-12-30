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
        walk = []
        start_node = random.choice(range(graph.number_of_nodes()))
        walk.append(start_node)

        while (not limit_length) or (len(walk) < max_length):
            neighbors = list(graph.neighbors(walk[-1]))
            unvisited_neighbors = list(set(neighbors) - set(walk))

            if unvisited_neighbors:
                next_node = random.choice(unvisited_neighbors)
                walk.append(next_node)
            else:
                # If there are no more neighbors to visit, end the walk
                break

        return walk

    def random_walks(
        self,
        graph: nx.Graph,
        num_walks: int,
        limit_length: bool = False,
        max_length: int = None  # type:ignore
    ) -> List[List[int]]:
        walks = []

        for _ in range(num_walks):
            walk = self.one_random_walk(
                graph,
                limit_length,
                max_length)
            walks.append(walk)

        return walks
