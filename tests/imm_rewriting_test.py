from __future__ import annotations
from io import StringIO
from optparse import Option
from pprint import pprint

from xdsl.dialects.builtin import *
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.std import *
from xdsl.dialects.arith import *
from xdsl.elevate import *
from xdsl.immutable_ir import *


def apply_strategy_and_compare(program: str, expected_program: str,
                               strategy: Strategy) -> ImmutableOperation:
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)

    parser = Parser(ctx, program)
    module: Operation = parser.parse_op()
    imm_module: ImmutableOperation = get_immutable_copy(module)

    rr = strategy.apply(imm_module)

    assert (rr.isSuccess() and isinstance(
        (resultOp := rr.result[-1]), ImmutableOperation))

    # for debugging
    printer = Printer()
    printer.print_op(resultOp._op)

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(resultOp._op)
    assert file.getvalue().strip() == expected_program.strip()

    return resultOp


def test_double_commute():
    """Tests a strategy which swaps the two operands of an arith.addi."""

    before = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  %2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
  %3 : !i32 = arith.constant() ["value" = 4 : !i32]
  %4 : !i32 = arith.addi(%2 : !i32, %3 : !i32)
  std.return(%4 : !i32)
}
"""

    once_commuted = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 4 : !i32]
  %1 : !i32 = arith.constant() ["value" = 1 : !i32]
  %2 : !i32 = arith.constant() ["value" = 2 : !i32]
  %3 : !i32 = arith.addi(%1 : !i32, %2 : !i32)
  %4 : !i32 = arith.addi(%0 : !i32, %3 : !i32)
  std.return(%4 : !i32)
}
"""

    @dataclass
    class CommuteAdd(Strategy):

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            if (isa(addOp := op, Addi)):
                return success(
                    ImmutableOperation.create_new(
                        Addi,
                        immutable_operands=[
                            addOp.operands[1], addOp.operands[0]
                        ],
                        result_types=[IntegerType.from_width(32)]))
            else:
                return failure("CommuteAdd")

    newModule = apply_strategy_and_compare(program=before,
                                           expected_program=once_commuted,
                                           strategy=topdown(CommuteAdd()))

    finalModule = apply_strategy_and_compare(program=once_commuted,
                                             expected_program=before,
                                             strategy=topdown(CommuteAdd()))


def testConstantFolding():
    before = \
"""module() {
%0 : !i32 = arith.constant() ["value" = 1 : !i32]
%1 : !i32 = arith.constant() ["value" = 2 : !i32]
%2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
%3 : !i32 = arith.constant() ["value" = 4 : !i32]
%4 : !i32 = arith.addi(%2 : !i32, %3 : !i32)
std.return(%4 : !i32)
}
"""

    twiceFolded = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 7 : !i32]
  std.return(%0 : !i32)
}
"""

    @dataclass
    class FoldConstantAdd(Strategy):

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            if (isa(addOp := op, Addi)) and (isa(
                    c1 := addOp.operands[0].get_op(), Constant)) and (isa(
                        c2 := addOp.operands[1].get_op(), Constant)):

                assert (isinstance((c1Attr := c1.get_attribute("value")).typ,
                                   IntegerType))
                assert (isinstance((c2Attr := c2.get_attribute("value")).typ,
                                   IntegerType))

                return success(
                    ImmutableOperation.create_new(
                        Constant,
                        result_types=[c1Attr.typ],
                        attributes={
                            "value":
                            IntegerAttr.from_params(
                                c1Attr.value.data + c2Attr.value.data,
                                c1Attr.typ)
                        }))
            else:
                return failure("FoldConstantAdd")

    apply_strategy_and_compare(program=before,
                               expected_program=twiceFolded,
                               strategy=seq(topdown(FoldConstantAdd()),
                                            topdown(FoldConstantAdd())))


if __name__ == "__main__":
    test_double_commute()
    testConstantFolding()