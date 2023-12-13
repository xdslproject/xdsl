from collections import Counter

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation


def count_number_of_operations(module: ModuleOp) -> dict[str, int]:
    """
    This function takes a ModuleOp and returns a dictionary containing the number of
    occurences of each Operation in the ModuleOp.
    """
    all_operations: list[Operation] = []

    for block in module.body.blocks:
        all_operations.extend(block.walk())

    # Apply .name to each element in the list using a list comprehension
    names_list = [op.name for op in all_operations]

    # Count occurrences of each unique Operation
    operation_counter = Counter(names_list)

    res_operations_count: dict[str, int] = dict(operation_counter.items())

    return res_operations_count
