from collections import Counter

from xdsl.dialects.builtin import ModuleOp


def count_number_of_operations(module: ModuleOp) -> dict[str, int]:
    """
    This function takes a ModuleOp and returns a dictionary containing the number of
    occurences of each Operation in the ModuleOp.
    """
    return Counter(op.name for op in module.walk())


def get_diff_operation_count(
    input_operation_count_tuple: tuple[tuple[str, int], ...],
    output_operation_count_tuple: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, int, str], ...]:
    """
    Function returning a tuple of tuples containing the diff of the input and output
    operation name and count.
    """
    input_op_count_dict = dict(input_operation_count_tuple)
    output_op_count_dict = dict(output_operation_count_tuple)
    all_keys = {*input_op_count_dict, *output_op_count_dict}

    res: dict[str, tuple[int, str]] = {}
    for k in all_keys:
        input_count = input_op_count_dict.get(k, 0)
        output_count = output_op_count_dict.get(k, 0)
        diff = output_count - input_count

        # convert diff to string
        if diff == 0:
            diff_str = "="
        elif diff > 0:
            diff_str = f"+{diff}"
        else:
            diff_str = str(diff)

        res[k] = (output_count, diff_str)

    return tuple((k, v0, v1) for (k, (v0, v1)) in sorted(res.items()))
