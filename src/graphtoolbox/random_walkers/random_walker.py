import random
import networkx as nx
from typing import List


class RandomWalker:
    def __init__(self) -> None:
        pass

    def one_random_walk(
        self,
        graph: nx.Graph,
        how_often_visited: List[int]
    ) -> List[int]:
        walk = []
        nodes = range(graph.number_of_nodes())
        start_node = nodes.index(min(nodes))
        walk.append(start_node)

        while True:
            neighbors = list(graph.neighbors(walk[-1]))
            unvisited_neighbors = list(set(neighbors) - set(walk))

            if unvisited_neighbors:
                #next_node = random.choice(unvisited_neighbors)
                next_node = random.choice(unvisited_neighbors)
                how_often_visited[next_node] += 1
                walk.append(next_node)
            else:
                # If there are no more neighbors to visit, end the walk
                break

        return walk

    def random_walks(
        self,
        graph: nx.Graph,
        num_walks: int
    ) -> List[List[int]]:
        walks = []

        how_often_visited = [0] * graph.number_of_nodes()

        for _ in range(num_walks):
            walk = self.one_random_walk(
                graph=graph,
                how_often_visited=how_often_visited)
            walks.append(walk)

        return walks
