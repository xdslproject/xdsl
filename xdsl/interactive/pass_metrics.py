from collections import Counter

from xdsl.dialects.builtin import ModuleOp


def count_number_of_operations(module: ModuleOp) -> dict[str, int]:
    """
    This function takes a ModuleOp and returns a dictionary containing the number of
    occurences of each Operation in the ModuleOp.
    """
    return Counter(op.name for op in module.walk())
