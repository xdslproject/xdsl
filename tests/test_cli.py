from xdsl.tools.command_line_tool import get_all_dialects, get_all_passes


def test_get_all_passes_names():
    for name, pass_factory in get_all_passes().items():
        assert name == pass_factory().name


def test_get_all_dialects_names():
    for name, dialect_factory in get_all_dialects().items():
        assert name == dialect_factory().name
