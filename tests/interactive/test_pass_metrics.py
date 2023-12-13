from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import (
    IndexType,
    IntegerAttr,
    ModuleOp,
)
from xdsl.interactive.pass_metrics import count_number_of_operations
from xdsl.ir import Block, MLContext, Region
from xdsl.parser import Parser
from xdsl.tools.command_line_tool import get_all_dialects

from xdsl.ir import Block, Region


def test_operation_counter():
    # build module
    index = IndexType()
    module = ModuleOp(Region([Block()]))
    with ImplicitBuilder(module.body):
        function = func.FuncOp("hello", ((index,), (index,)))
        with ImplicitBuilder(function.body) as (n,):
            two = arith.Constant(IntegerAttr(2, index)).result
            three = arith.Constant(IntegerAttr(2, index)).result
            res_1 = arith.Muli(n, two)
            res_2 = arith.Muli(n, three)
            res = arith.Muli(res_1, res_2)
            func.Return(res)

    expected_res = {
        "func.func": 1,
        "arith.constant": 2,
        "arith.muli": 3,
        "func.return": 1,
        "builtin.module": 1,
    }

    res = count_number_of_operations(module)
    assert res == expected_res


def test_operation_counter_with_parsing_text():
    text = """builtin.module {
  func.func @hello(%n : index) -> index {
    %two = arith.constant 2 : index
    %res = arith.muli %n, %two : index
    func.return %res : index
  }
}
"""

    ctx = MLContext(True)
    for dialect in get_all_dialects():
        ctx.load_dialect(dialect)
    parser = Parser(ctx, text)
    module = parser.parse_module()

    expected_res = {
        "func.func": 1,
        "arith.constant": 1,
        "arith.muli": 1,
        "func.return": 1,
        "builtin.module": 1,
    }

    res = count_number_of_operations(module)
    assert res == expected_res
