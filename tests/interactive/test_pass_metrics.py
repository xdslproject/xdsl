from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, func, get_all_dialects
from xdsl.dialects.builtin import (
    IndexType,
    IntegerAttr,
    ModuleOp,
)
from xdsl.interactive.pass_metrics import (
    count_number_of_operations,
    get_diff_operation_count,
)
from xdsl.ir import Block, Region
from xdsl.parser import Parser


def test_operation_counter():
    # build module
    index = IndexType()
    module = ModuleOp(Region([Block()]))
    with ImplicitBuilder(module.body):
        function = func.FuncOp("hello", ((index,), (index,)))
        with ImplicitBuilder(function.body) as (n,):
            two = arith.ConstantOp(IntegerAttr(2, index)).result
            three = arith.ConstantOp(IntegerAttr(2, index)).result
            res_1 = arith.MuliOp(n, two)
            res_2 = arith.MuliOp(n, three)
            res = arith.MuliOp(res_1, res_2)
            func.ReturnOp(res)

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

    ctx = Context(True)
    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)
    parser = Parser(ctx, text)
    module = parser.parse_module()

    expected_res = {
        "arith.constant": 1,
        "arith.muli": 1,
        "builtin.module": 1,
        "func.func": 1,
        "func.return": 1,
    }

    res = count_number_of_operations(module)
    assert res == expected_res


def test_get_diff_operation_count():
    # get input module
    input_text = """builtin.module {
  func.func @hello(%n : index) -> index {
    %two = arith.constant 2 : index
    %res = arith.muli %n, %two : index
    func.return %res : index
  }
}
"""

    ctx = Context(True)
    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)
    parser = Parser(ctx, input_text)
    input_module = parser.parse_module()

    # get output module
    output_text = """builtin.module {
  func.func @hello(%n : index) -> index {
    %two = riscv.li 2 : !riscv.reg
    %two_1 = builtin.unrealized_conversion_cast %two : !riscv.reg to index
    %res = builtin.unrealized_conversion_cast %n : index to !riscv.reg
    %res_1 = builtin.unrealized_conversion_cast %two_1 : index to !riscv.reg
    %res_2 = riscv.mul %res, %res_1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %res_3 = builtin.unrealized_conversion_cast %res_2 : !riscv.reg to index
    func.return %res_3 : index
  }
}
"""
    parser = Parser(ctx.clone(), output_text)
    output_module = parser.parse_module()

    expected_diff_res: tuple[tuple[str, int, str], ...] = (
        ("arith.constant", 0, "-1"),
        ("arith.muli", 0, "-1"),
        ("builtin.module", 1, "="),
        ("builtin.unrealized_conversion_cast", 4, "+4"),
        ("func.func", 1, "="),
        ("func.return", 1, "="),
        ("riscv.li", 1, "+1"),
        ("riscv.mul", 1, "+1"),
    )

    assert expected_diff_res == get_diff_operation_count(
        tuple(count_number_of_operations(input_module).items()),
        tuple(count_number_of_operations(output_module).items()),
    )
