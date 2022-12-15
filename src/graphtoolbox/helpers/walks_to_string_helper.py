import re


class WalksToStringHelper:
    def __init__(self) -> None:
        pass

    def walks_to_string(self, walks):
        SENTENCE_END_SYMBOL = ' END '
        WORD_END_SYMBOL = ' '

        the_string = SENTENCE_END_SYMBOL.join(
            [WORD_END_SYMBOL.join([f'_{str(num)}_' for num in walk]) for walk in walks])

        return the_string

    def get_replace_dict(self, graph, attributed):
        replace_dict = dict()

        if attributed:
            for node in range(graph.number_of_nodes()):
                replace_dict['_' + str(node) + '_'] = ','.join([str(num)
                                                                for num in graph.nodes[node]['feature']])
        else:
            for node in range(graph.number_of_nodes()):
                replace_dict['_' + str(node) + '_'] = str(graph.degree[node])

        return replace_dict

    def get_document(self, walks, replace_dict):
        walks = self.walks_to_string(walks)

        pattern = '|'.join(sorted(re.escape(k) for k in replace_dict))

        the_better_string = re.sub(pattern, lambda m: replace_dict.get(
            m.group(0).upper()), walks, flags=re.IGNORECASE)

        return (the_better_string)
