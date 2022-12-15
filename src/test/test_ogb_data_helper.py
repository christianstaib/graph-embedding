from graphtoolbox import ogb_data_helper


def test_dummy_function():
    gh = ogb_data_helper.OgbDataHelper()
    return_value = gh.dummy_function()
    assert return_value == 1
