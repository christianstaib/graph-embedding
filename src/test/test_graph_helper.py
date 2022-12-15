from graphtoolbox import graph_helper


def test_dummy_function():
    gh = graph_helper.GraphHelper()
    return_value = gh.dummy_function()
    assert return_value == 1
