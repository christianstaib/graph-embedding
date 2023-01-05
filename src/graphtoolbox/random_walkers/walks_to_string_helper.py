import re


class WalksToStringHelper:
    def __init__(self) -> None:
        pass

    def walks_to_strings(self, walks):
        strings = []

        for walk in walks:
            walk_as_string = ' '.join([f'_{str(vertex)}_' for vertex in walk])
            strings.append(walk_as_string)

        return strings

    def get_replace_dict(self, graph):
        replace_dict = dict()

        for node in range(graph.number_of_nodes()):
            key = '_' + str(node) + '_'
            value = ','.join([str(num) for num in graph.nodes[node]['feature']])

            replace_dict[key] = value

        return replace_dict

    def get_documents(self, strings, replace_dict):
        strings = self.walks_to_strings(strings)

        documents = []

        for string in strings:
            for key, value in replace_dict.items():
                string = re.sub(key, value, string)
            documents.append(string)

        return documents
