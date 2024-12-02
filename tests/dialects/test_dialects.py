from xdsl.dialects import get_all_dialects


def test_op_class_names():
    """
    Make sure that all operation class names match our convention of having an "Op"
    suffix.
    """
    all_dialects = get_all_dialects()
    malformed_op_names = tuple(
        (op.name, op.__name__)
        for dialect_factory in all_dialects.values()
        for op in dialect_factory().operations
        if op.__name__[-2:] != "Op"
    )

    assert not malformed_op_names
