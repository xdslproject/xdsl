"""
Test the usage of fsm dialect.
"""


from xdsl.dialects import builtin
from xdsl.dialects.fsm import Machine, State
from xdsl.ir.core import Block, Region


def test_output_types_not_consistent():
    inputs = [builtin.IndexType()]

    # Create Blocks and Regions

    arg = builtin.ArrayAttr([builtin.DictionaryAttr({"in_d": builtin.i32})])
    res = builtin.ArrayAttr([builtin.DictionaryAttr({"out_d": builtin.i32})])
    arg_name = builtin.ArrayAttr([builtin.StringAttr("in")])
    res_name = builtin.ArrayAttr([builtin.StringAttr("out")])

    m = Machine(
        body=Region(Block([])),
        sym_name="test_machine",
        initial_state="test_state_0",
        function_type=builtin.FunctionType.from_lists(inputs, []),
        arg_attrs=arg,
        res_attrs=res,
        arg_names=arg_name,
        res_names=res_name,
    )

    s = State(sym_name="test_state_0")

    m.verify()
    s.verify()
