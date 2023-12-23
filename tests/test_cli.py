from xdsl.tools.command_line_tool import get_all_passes


def test_get_all_passes_names():
    for name, pass_factory in get_all_passes():
        assert name == pass_factory().name
