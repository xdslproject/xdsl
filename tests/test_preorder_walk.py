import pytest

from xdsl.context import MLContext
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.func import Func, FuncOp
from xdsl.dialects.scf import For, If, Scf
from xdsl.parser import Parser

test_prog = """
  "func.func"() <{function_type = (i1, i32, i32) -> i32, sym_name = "example_func"}> ({
  ^bb0(%arg0: i1, %arg1: i32, %arg2: i32):
    %0 = "scf.if"(%arg0) ({
      %1 = "arith.constant"() <{value = 42 : i32}> : () -> i32
      %2 = "arith.constant"() <{value = true}> : () -> i1
      %3 = "scf.if"(%2) ({
        %4 = "arith.constant"() <{value = 84 : i32}> : () -> i32
        "scf.yield"(%4) : (i32) -> ()
      }, {
        %4 = "arith.constant"() <{value = 21 : i32}> : () -> i32
        "scf.yield"(%4) : (i32) -> ()
      }) : (i1) -> i32
      "scf.yield"(%3) : (i32) -> ()
    }, {
      %1 = "arith.index_cast"(%arg1) : (i32) -> index
      %2 = "arith.index_cast"(%arg2) : (i32) -> index
      %3 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      "scf.for"(%1, %2, %1) ({
      ^bb0(%arg3: index):
        %4 = "arith.index_cast"(%arg3) : (index) -> i32
        %5 = "arith.constant"() <{value = 10 : i32}> : () -> i32
        %6 = "arith.constant"() <{value = false}> : () -> i1
        "scf.if"(%6) ({
          %7 = "arith.constant"() <{value = 100 : i32}> : () -> i32
          "scf.yield"() : () -> ()
        }, {
          %7 = "arith.constant"() <{value = 200 : i32}> : () -> i32
          "scf.yield"() : () -> ()
        }) : (i1) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"(%3) : (i32) -> ()
    }) : (i1) -> i32
    "func.return"(%0) : (i32) -> ()
  }) : () -> ()
"""


def test_preorder_walk():
    ctx = MLContext()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Arith)
    ctx.load_dialect(Func)
    ctx.load_dialect(Scf)

    parser = Parser(ctx, test_prog)
    op = parser.parse_op()

    assert isinstance(op, FuncOp)

    first_if = op.body.block.ops.first
    assert isinstance(first_if, If)
    second_if = list(first_if.true_region.block.ops)[2]
    assert isinstance(second_if, If)
    for_loop = list(first_if.false_region.block.ops)[3]
    assert isinstance(for_loop, For)
    third_if = list(for_loop.body.block.ops)[3]
    assert isinstance(third_if, If)

    it = op.walk_blocks_preorder()
    assert next(it) == op.body.block
    assert next(it) == first_if.true_region.block
    assert next(it) == second_if.true_region.block
    assert next(it) == second_if.false_region.block
    assert next(it) == first_if.false_region.block
    assert next(it) == for_loop.body.block
    assert next(it) == third_if.true_region.block
    assert next(it) == third_if.false_region.block

    with pytest.raises(StopIteration):
        next(it)
