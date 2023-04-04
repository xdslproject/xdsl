from typing import cast
import pytest

from xdsl.ir import MLContext, Operation, Block, Region, ErasedSSAValue
from xdsl.dialects.arith import Addi, Subi, Constant
from xdsl.dialects.builtin import IntegerType, i32, IntegerAttr, ModuleOp
from xdsl.dialects.scf import If
from xdsl.parser import XDSLParser
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.func import Func
from xdsl.dialects.arith import Arith
from xdsl.dialects.cf import Cf


def test_ops_accessor():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    # Operation to add these constants
    c = Addi.get(a, b)

    block0 = Block.from_ops([a, b, c])
    # Create a region to include a, b, c
    region = Region.from_block_list([block0])

    assert len(region.ops) == 3
    assert len(region.blocks[0].ops) == 3

    # Operation to subtract b from a
    d = Subi.get(a, b)

    assert d.results[0] != c.results[0]

    assert c.lhs.owner is a
    assert c.rhs.owner is b
    assert d.lhs.owner is a
    assert d.rhs.owner is b


def test_ops_accessor_II():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(2, i32)
    # Operation to add these constants
    c = Addi.get(a, b)

    block0 = Block.from_ops([a, b, c])
    # Create a region to include a, b, c
    region = Region.from_block_list([block0])

    assert len(region.ops) == 3
    assert len(region.blocks[0].ops) == 3

    # Operation to subtract b from a
    d = Subi.get(a, b)

    assert d.results[0] != c.results[0]

    # Erase operations and block
    region2 = Region()
    region.move_blocks(region2)

    region2.blocks[0].erase_op(a, safe_erase=False)
    region2.blocks[0].erase_op(b, safe_erase=False)
    region2.blocks[0].erase_op(c, safe_erase=False)

    assert isinstance(c.lhs, ErasedSSAValue)
    assert isinstance(c.rhs, ErasedSSAValue)
    assert c.lhs.owner is a
    assert c.rhs.owner is b

    region2.detach_block(block0)
    region2.drop_all_references()

    assert len(region2.blocks) == 0


def test_ops_accessor_III():
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
    block2 = Block.from_ops([])

    region0 = Region.from_block_list([block0, block1])
    region1 = Region.from_block_list([block2])

    with pytest.raises(ValueError):
        region0.ops

    with pytest.raises(ValueError):
        region0.op

    with pytest.raises(ValueError):
        region1.op

    with pytest.raises(Exception):
        region1.detach_block(block0)

    region0.detach_block(block0)
    region0.detach_block(0)
    with pytest.raises(IndexError):
        region0.detach_block(1)


def test_op_clone():
    a = Constant.from_int_and_width(1, 32)
    b = a.clone()

    assert a is not b

    assert isinstance(b.value, IntegerAttr)
    b_value = cast(IntegerAttr[IntegerType], b.value)

    assert b_value.value.data == 1
    assert b_value.typ.width.data == 32


def test_op_clone_with_regions():
    cond = Constant.from_int_and_width(1, 1)
    a = Constant.from_int_and_width(1, 32)
    if_ = If.get(cond, [], Region.from_operation_list([a]),
                 Region.from_operation_list([a.clone()]))

    if2 = if_.clone()

    assert if2 is not if_
    assert len(if2.true_region.ops) == 1
    assert len(if2.false_region.ops) == 1
    assert if2.true_region.op is not if_.true_region.op
    assert if2.false_region.op is not if_.false_region.op


##################### Testing is_structurally_equal #####################

program_region = \
"""builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
}
"""
program_region_2 = \
"""builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 2 : !i32]
}
"""
program_region_2_diff_name = \
"""builtin.module() {
  %cst : !i32 = arith.constant() ["value" = 2 : !i32]
}
"""
program_region_2_diff_type = \
"""builtin.module() {
  %0 : !i64 = arith.constant() ["value" = 2 : !i64]
}
"""
program_add = \
"""builtin.module() {
%0 : !i32 = arith.constant() ["value" = 1 : !i32]
%1 : !i32 = arith.constant() ["value" = 2 : !i32]
%2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
}
"""
program_add_2 = \
"""builtin.module() {
%0 : !i32 = arith.constant() ["value" = 1 : !i32]
%1 : !i32 = arith.constant() ["value" = 2 : !i32]
%2 : !i32 = arith.addi(%1 : !i32, %0 : !i32)
}
"""
program_func = \
"""builtin.module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
    func.return(%2 : !i32)
  }
}
"""
program_successors = \
"""
    func.func() ["sym_name" = "unconditional_br", "function_type" = !fun<[], []>, "sym_visibility" = "private"] {
    ^0:
        cf.br() (^1)
    ^1:
        cf.br() (^0)
    }
"""


@pytest.mark.parametrize(
    "args, expected_result",
    [([program_region, program_region], True),
     ([program_region_2, program_region_2], True),
     ([program_region_2_diff_type, program_region_2_diff_type], True),
     ([program_region_2_diff_name, program_region_2_diff_name], True),
     ([program_region, program_region_2], False),
     ([program_region_2, program_region_2_diff_type], False),
     ([program_region_2, program_region_2_diff_name], True),
     ([program_add, program_add], True),
     ([program_add_2, program_add_2], True),
     ([program_add, program_add_2], False),
     ([program_func, program_func], True),
     ([program_successors, program_successors], True),
     ([program_successors, program_func], False),
     ([program_successors, program_add], False)])
def test_is_structurally_equivalent(args: list[str], expected_result: bool):
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Func)
    ctx.register_dialect(Arith)
    ctx.register_dialect(Cf)

    parser = XDSLParser(ctx, args[0])
    lhs: Operation = parser.parse_op()

    parser = XDSLParser(ctx, args[1])
    rhs: Operation = parser.parse_op()

    assert lhs.is_structurally_equivalent(rhs) == expected_result


def test_is_structurally_equivalent_incompatible_ir_nodes():
    program_func = \
  """builtin.module() {
    func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
    ^0(%0 : !i32, %1 : !i32):
      %2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
      %3 : !i32 = arith.constant() ["value" = 2 : !i32]
      func.return(%3 : !i32)
    ^1(%4 : !i32, %5 : !i32):
      func.return(%4 : !i32)
    }
  }
  """
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Func)
    ctx.register_dialect(Arith)
    ctx.register_dialect(Cf)

    parser = XDSLParser(ctx, program_func)
    program = parser.parse_operation()

    assert isinstance(program, ModuleOp)

    assert program.is_structurally_equivalent(program.regions[0]) == False
    assert program.is_structurally_equivalent(
        program.regions[0].blocks[0]) == False
    assert program.regions[0].is_structurally_equivalent(program) == False
    assert program.regions[0].blocks[0].is_structurally_equivalent(
        program) == False
    assert program.ops[0].regions[0].blocks[0].ops[
        0].is_structurally_equivalent(
            program.ops[0].regions[0].blocks[0].ops[1]) == False
    assert program.ops[0].regions[0].blocks[0].is_structurally_equivalent(
        program.ops[0].regions[0].blocks[1]) == False


def test_descriptions():
    a = Constant.from_int_and_width(1, 32)

    assert str(a.value) == '1 : !i32'
    assert f'{a.value}' == '1 : !i32'

    assert str(a) == '%0 : !i32 = arith.constant() ["value" = 1 : !i32]'
    assert f'{a}' == 'Constant(%0 : !i32 = arith.constant() ["value" = 1 : !i32])'

    m = ModuleOp.from_region_or_ops([a])

    assert str(m) == '''\
builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
}'''

    assert f'{m}' == '''\
ModuleOp(
\tbuiltin.module() {
\t  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
\t}
)'''


def test_op_custom_verify_is_done_last():
    a = Constant.from_int_and_width(1, i32)
    b = Constant.from_int_and_width(1, i32)
    error = Exception("CustomException")

    def verify_():
        raise error
    a.__dict__['verify_'] = verify_
    a.attributes.pop('value')
    b.attributes.pop('value')
    try:
        a.verify()
    except Exception as e_a:
        assert e_a is not error
        try:
            b.verify()
        except Exception as e_b:
            assert e_a.args == e_b.args
            assert e_a.args == ("attribute value expected",)
