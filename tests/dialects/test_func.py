import pytest

from conftest import assert_print_op
from xdsl.dialects.func import FuncOp, Return, Call
from xdsl.dialects.arith import Addi, Constant
from xdsl.dialects.builtin import IntegerAttr, i32, ModuleOp, i64
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import VerifyException


def test_func():
    # This test creates two FuncOps with different approaches that
    # represent the same code and checks their structure
    # Create two constants and add them, add them in a region and
    # create a function
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    # Operation to add these constants
    c = Addi.get(a, b)

    # Create a region to include a, b, c
    region = Region.from_operation_list([a, b, c])

    # Use this region to create a func0
    func0 = FuncOp.from_region("func0", [], [], region)

    # Alternative generation of func0
    func1 = FuncOp.from_region(
        "func1", [], [],
        Region.from_operation_list([
            a := Constant.from_int_and_width(1, i32),
            b := Constant.from_int_and_width(2, i32),
            Addi.get(a, b),
        ]))

    assert len(func0.regions[0].ops) == 3
    assert len(func1.regions[0].ops) == 3
    assert type(func0.regions[0].ops[0]) is Constant
    assert type(func1.regions[0].ops[0]) is Constant
    assert type(func0.regions[0].ops[1]) is Constant
    assert type(func1.regions[0].ops[1]) is Constant
    assert type(func0.regions[0].ops[2]) is Addi


def test_func_II():
    # Create constants `from_attr` and add them, add them in blocks, blocks in
    # a region and create a function
    a = Constant.from_attr(IntegerAttr.from_int_and_width(1, 32), i32)
    b = Constant.from_attr(IntegerAttr.from_int_and_width(2, 32), i32)
    c = Constant.from_attr(IntegerAttr.from_int_and_width(3, 32), i32)
    d = Constant.from_attr(IntegerAttr.from_int_and_width(4, 32), i32)

    # Operation to add these constants
    e = Addi.get(a, b)
    f = Addi.get(c, d)

    # Create Blocks and Regions
    block0 = Block.from_ops([a, b, e])
    block1 = Block.from_ops([c, d, f])
    region0 = Region.from_block_list([block0, block1])

    # Use this region to create a func0
    func1 = FuncOp.from_region("func1", [], [], region0)

    assert len(func1.regions[0].blocks[0].ops) == 3
    assert len(func1.regions[0].blocks[1].ops) == 3
    assert type(func1.regions[0].blocks[0].ops[0]) is Constant
    assert type(func1.regions[0].blocks[0].ops[1]) is Constant
    assert type(func1.regions[0].blocks[0].ops[2]) is Addi
    assert type(func1.regions[0].blocks[1].ops[0]) is Constant
    assert type(func1.regions[0].blocks[1].ops[1]) is Constant
    assert type(func1.regions[0].blocks[1].ops[2]) is Addi


def test_wrong_blockarg_types():
    r = Region.from_block_list(
        [Block.from_callable([i32], lambda x: [Addi.get(x, x)])])
    f = FuncOp.from_region("f", [i32, i32], [], r)
    with pytest.raises(VerifyException) as e:
        f.verify()

    assert e.value.args[0] == (
        "Expected entry block arguments to have the same "
        "types as the function input types")


def test_callable_constructor():
    f = FuncOp.from_callable("f", [], [], lambda: [])
    assert f.sym_name.data == "f"
    assert f.body.ops == []


def test_call():
    """
    Pass two integers to a function and return their sum
    """
    # Create two constants and add them, then return
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)

    # Create a block using the types of a, b
    block0 = Block.from_arg_types([a.result.typ, b.result.typ])
    # Create a Addi operation to use the args of the block
    c = Addi.get(block0.args[0], block0.args[1])
    # Create a return operation and add it in the block
    ret0 = Return.get(c)
    block0.add_ops([c, ret0])
    # Create a region with the block
    region = Region.from_block_list([block0])

    # Create a func0 that gets the block args as arguments, returns the resulting
    # type of c and has the region as body
    func0 = FuncOp.from_region("func0",
                               [block0.args[0].typ, block0.args[1].typ],
                               [c.result.typ], region)

    # Create a call for this function, passing a, b as args
    # and returning the type of the return
    call0 = Call.get(func0.sym_name.data, [a, b], [ret0.arguments[0].typ])

    # Wrap all in a ModuleOp
    mod = ModuleOp.from_region_or_ops([func0, a, b, call0])

    expected = \
        """
"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : i32, %1 : i32):
    %2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
    "func.return"(%2) : (i32) -> ()
  }) {"sym_name" = "func0", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
  %3 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %4 = "arith.constant"() {"value" = 2 : i32} : () -> i32
  %5 = "func.call"(%3, %4) {"callee" = @func0} : (i32, i32) -> i32
}) : () -> ()
"""  # noqa
    assert len(call0.operands) == 2
    assert len(func0.operands) == 0

    assert_print_op(mod, expected, None)


def test_call_II():
    """
    Pass an integer a to a function and return a + a
    """
    # Create two constants and add them, then return
    a = Constant.from_int_and_width(1, i32)

    # Create a block using the type of a
    block0 = Block.from_arg_types([a.result.typ])
    # Create a Addi operation to use the args of the block
    c = Addi.get(block0.args[0], block0.args[0])
    # Create a return operation and add it in the block
    ret0 = Return.get(c)
    block0.add_ops([c, ret0])
    # Create a region with the block
    region = Region.from_block_list([block0])

    # Create a func0 that gets the block args as arguments, returns the resulting
    # type of c and has the region as body
    func0 = FuncOp.from_region("func1", [block0.args[0].typ], [c.result.typ],
                               region)

    # Create a call for this function, passing a, b as args
    # and returning the type of the return
    call0 = Call.get(func0.sym_name.data, [a], [ret0.arguments[0].typ])

    # Wrap all in a ModuleOp
    mod = ModuleOp.from_region_or_ops([func0, a, call0])

    expected = \
        """
"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : i32):
    %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {"sym_name" = "func1", "function_type" = (i32) -> i32, "sym_visibility" = "private"} : () -> ()
  %2 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %3 = "func.call"(%2) {"callee" = @func1} : (i32) -> i32
}) : () -> ()
"""  # noqa
    assert len(call0.operands) == 1
    assert len(func0.operands) == 0

    assert_print_op(mod, expected, None)


def test_return():
    # Create two constants and add them, then return
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    c = Constant.from_int_and_width(3, i32)

    # Use these operations to create a Return operation
    ret0 = Return.get(a, b, c)
    assert len(ret0.operands) == 3


def test_external_func_def():
    # FuncOp.external must produce a function with an empty body
    ext = FuncOp.external("testname", [i32, i32], [i64])

    assert len(ext.regions) == 1
    assert len(ext.regions[0].ops) == 0
    assert ext.sym_name.data == "testname"
