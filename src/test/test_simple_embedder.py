from graphtoolbox import simple_embedder


def test_dummy_function():
    gh = simple_embedder.SimpleEmbedder()
    return_value = gh.dummy_function()
    assert return_value == 1
