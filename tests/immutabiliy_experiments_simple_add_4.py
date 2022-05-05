from __future__ import annotations
from io import StringIO
from optparse import Option
from pprint import pprint
import xdsl.dialects.arith as arith
import xdsl.dialects.builtin as builtin
import xdsl.dialects.scf as scf
from xdsl.dialects.affine import Affine
from xdsl.parser import Parser
from xdsl.pattern_rewriter import PatternRewriteWalker, PatternRewriter, RewritePattern
from xdsl.printer import Printer
from xdsl.dialects.func import *
from xdsl.dialects.func import Return as stdReturn

from xdsl.elevate import *
from xdsl.immutable_ir import *

import difflib

###
#
#   This is not a test. It is a file for prototyping and experimentation. To be removed in a non draft PR
#
###


def rewriting_with_immutability_experiments():
    # constant folding
    before = \
"""module() {
%0 : !i32 = arith.constant() ["value" = 1 : !i32]
%1 : !i32 = arith.constant() ["value" = 2 : !i32]
%2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
%3 : !i32 = arith.constant() ["value" = 4 : !i32]
%4 : !i32 = arith.addi(%2 : !i32, %3 : !i32)
func.return(%4 : !i32)
}
"""

    block_args_before = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32,!i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0: !i32, %1: !i32):
    %2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
    func.return(%2 : !i32)
  }
}
"""

    # In current xdsl I have no way to get to the function from the call
    not_possible = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32,!i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0: !i32, %1: !i32):
    %3 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
    func.return(%3 : !i32)
  }
  %4 : !i32 = arith.constant() ["value" = 0 : !i32]
  %5 : !i32 = arith.constant() ["value" = 1 : !i32]
  %6 : !i32 = func.call(%4 : !i32, %5 : !i32) ["callee" = @test] 
}
"""

    before_scf_if = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32,!i32], [!i32]>, "sym_visibility" = "private"] {
  ^0():
    %0 : !i1 = arith.constant() ["value" = 1 : !i1]
    %1 : !i32 = scf.if(%0 : !i1) {
      %2 : !i32 = arith.constant() ["value" = 0 : !i32]
      scf.yield(%2 : !i32)
    }
    func.return(%1 : !i32)
  }
}
"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 7 : !i32]
  func.return(%0 : !i32)
}
"""

    @dataclass
    class FoldConstantAdd(Strategy):

        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(
                    op_type=arith.Addi,
                    operands=[IVal(op=IOp(op_type=arith.Constant, 
                                            attributes={"value": IntegerAttr() as attr1}) as c1), 
                              IVal(op=IOp(op_type=arith.Constant, 
                                            attributes={"value": IntegerAttr() as attr2}))]):
                    result = from_op(c1,
                            attributes={
                                "value":
                                IntegerAttr.from_params(
                                    attr1.value.data + attr2.value.data,
                                    attr1.typ)
                            })
                    return success(result)
                case _:
                    return failure(self)

    @dataclass
    class CommuteAdd(Strategy):

        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(op_type=arith.Addi,
                        operands=[operand0, operand1]):
                    result = from_op(op, operands=[operand1, operand0])
                    return success(result)
                case _:
                    return failure(self)

    @dataclass
    class ChangeConstantTo42(Strategy):

        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(op_type=arith.Constant, attributes={"value": IntegerAttr() as attr}):
                    # TODO: this should not be asserted but matched above
                    assert isinstance(attr, IntegerAttr)
                    result = from_op(op,
                                attributes={
                                    "value":
                                    IntegerAttr.from_params(42,
                                                            attr.typ)
                                })
                    return success(result)
                case _:
                    return failure(self)

    @dataclass
    class InlineIf(Strategy):

        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(op_type=scf.If,
                            operands=[IRes(op=IOp(op_type=arith.Constant, attributes={"value": IntegerAttr(value=IntAttr(data=1))}))],
                            region=IRegion(block=
                                IBlock(ops=[*_, IOp(op_type=scf.Yield, operands=[IRes(op=returned_op)])]))):                         
                            return success(returned_op)
                case _:
                    return failure(self)

    @dataclass
    class AddZero(Strategy):

        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(results=[IRes(typ=IntegerType() as type)]):                    
                    result = new_op(Addi, [
                        op,
                        new_op(Constant,
                            attributes={
                                "value": IntegerAttr.from_int_and_width(0, 32)
                            }, result_types=[type])
                    ], result_types=[type])

                    return success(result)
                case _:
                    return failure(self)

    ctx = MLContext()
    Builtin(ctx)
    Func(ctx)
    Arith(ctx)
    scf.Scf(ctx)
    Affine(ctx)

    parser = Parser(ctx, before_scf_if)
    beforeM: Operation = parser.parse_op()
    immBeforeM: IOp = get_immutable_copy(beforeM)

    print("before:")
    printer = Printer()
    printer.print_op(beforeM)
    # test = topdown(seq(debug(), fail())).apply(immBeforeM)

    print("mutable_copy:")
    printer = Printer()
    printer.print_op(immBeforeM.get_mutable_copy())

    rrImmM1 = topdown(AddZero()).apply(immBeforeM)
    print(rrImmM1)
    assert rrImmM1.isSuccess()

    printer = Printer()
    printer.print_op(rrImmM1.result_op.get_mutable_copy())

    # rrImmM2 = topdown(FoldConstantAdd()).apply(rrImmM1.result[-1])
    # print(rrImmM2)
    # assert (rrImmM2.isSuccess()
    #         and isinstance(rrImmM2.result[-1], IOp))

    # file = StringIO("")
    # printer = Printer(stream=file)
    # printer.print_op(rrImmM2.result[-1].get_mutable_copy())

    # For debugging: printing the actual output
    # print("after:")
    # print(file.getvalue().strip())

    checkDiff = False
    if checkDiff:
        diff = difflib.Differ().compare(file.getvalue().splitlines(True),
                                        expected.splitlines(True))
        print(''.join(diff))
        assert file.getvalue().strip() == expected.strip()


if __name__ == "__main__":
    rewriting_with_immutability_experiments()
