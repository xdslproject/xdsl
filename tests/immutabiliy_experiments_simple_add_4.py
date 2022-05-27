from __future__ import annotations
from io import StringIO
from xdsl.dialects import memref
import xdsl.dialects.arith as arith
import xdsl.dialects.builtin as builtin
import xdsl.dialects.scf as scf
import xdsl.dialects.affine as affine
import xdsl.dialects.func as func
from xdsl.parser import Parser
from xdsl.printer import Printer

from xdsl.elevate import *
import xdsl.elevate as elevate
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
%unused : !i32 = arith.constant() ["value" = 42 : !i32]
%0 : !i32 = arith.constant() ["value" = 1 : !i32]
%1 : !i32 = arith.constant() ["value" = 2 : !i32]
%2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
%3 : !i32 = arith.constant() ["value" = 4 : !i32]
%4 : !i32 = arith.addi(%2 : !i32, %3 : !i32)
func.return(%4 : !i32)
}
"""

    before_affine_loops_3_deep_nest = \
"""module() {
  func.func() ["sym_name" = "affine_mm", "function_type" = !fun<[!memref<[256 : !index, 256 : !index], !i32>, !memref<[256 : !index, 256 : !index], !i32>, !memref<[256 : !index, 256 : !index], !i32>], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !memref<[256 : !index, 256 : !index], !i32>, %1 : !memref<[256 : !index, 256 : !index], !i32>, %2 : !memref<[256 : !index, 256 : !index], !i32>):
    %res_outer : !memref<[256 : !index, 256 : !index], !i32> = scf.for() ["lower_bound" = 0 : !index, "upper_bound" = 256 : !index, "step" = 1 : !index] {
    ^1(%3 : !index):
      %res_middle : !memref<[256 : !index, 256 : !index], !i32> = scf.for() ["lower_bound" = 0 : !index, "upper_bound" = 256 : !index, "step" = 1 : !index] {
      ^2(%4 : !index):
        scf.for() ["lower_bound" = 0 : !index, "upper_bound" = 256 : !index, "step" = 1 : !index] {
        ^3(%7 : !index):
          %9 : !i32 = memref.load(%0 : !memref<[256 : !index, 256 : !index], !i32>, %3 : !index, %7 : !index)
          %10 : !i32 = memref.load(%1 : !memref<[256 : !index, 256 : !index], !i32>, %7 : !index, %4 : !index)
          %11 : !i32 = memref.load(%2 : !memref<[256 : !index, 256 : !index], !i32>, %3 : !index, %4 : !index)
          %12 : !i32 = arith.mulf(%9 : !i32, %10 : !i32)
          %13 : !i32 = arith.addf(%11 : !i32, %12 : !i32)
          memref.store(%13 : !i32, %2 : !memref<[256 : !index, 256 : !index], !i32>, %3 : !index, %4 : !index)
        }
      }
    }
    func.return(%res_outer : !memref<[256 : !index, 256 : !index], !i32>)
  }
}
"""


    before_affine_loops = \
"""module() {
  func.func() ["sym_name" = "sum_vec", "function_type" = !fun<[!memref<[128 : !index], !i32>], [!i32]>, "sym_visibility" = "private"] {
  ^0(%ref : !memref<[128 : !index], !i32>):
    %const0 : !i32 = arith.constant() ["value" = 0 : !i32]
    %r : !i32 = affine.for() ["lower_bound" = 0 : !index, "upper_bound" = 256 : !index, "step" = 1 : !index] {
    ^11(%i : !index):
      %val : !i32 = memref.load(%ref : !memref<[128 : !index], !i32>, %i : !index)
      %res : !i32 = arith.addi(%const0 : !i32, %val : !i32)
      affine.yield(%res : !i32)
    }
    func.return(%r : !i32)
  }
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

    @dataclass(frozen=True)
    class FoldConstantAdd(Strategy):

        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(
                    op_type=arith.Addi,
                    operands=[ISSAValue(op=IOp(op_type=arith.Constant, 
                                            attributes={"value": IntegerAttr() as attr1}) as c1), 
                              ISSAValue(op=IOp(op_type=arith.Constant, 
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

    @dataclass(frozen=True)
    class CommuteAdd(Strategy):

        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(op_type=arith.Addi,
                        operands=[operand0, operand1]):
                    result = from_op(op, operands=[operand1, operand0])
                    return success(result)
                case _:
                    return failure(self)

    @dataclass(frozen=True)
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

    @dataclass(frozen=True)
    class InlineIf(Strategy):

        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(op_type=scf.If,
                            operands=[IResult(op=IOp(op_type=arith.Constant, attributes={"value": IntegerAttr(value=IntAttr(data=1))}))],
                            region=IRegion(block=
                                IBlock(ops=[*_, IOp(op_type=scf.Yield, operands=[IResult(op=returned_op)])]))):                         
                            return success(returned_op)
                case _:
                    return failure(self)

    @dataclass(frozen=True)
    class AddZero(Strategy):

        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(results=[IResult(typ=IntegerType() as type)]):                    
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


    @dataclass(frozen=True)
    class LoopUnroll(Strategy):

        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(op_type=affine.For, results=[IResult(typ=IntegerType() as type)], 
                            attributes={"lower_bound": IntegerAttr(value=IntAttr(data=lb)), 
                                        "upper_bound": IntegerAttr(value=IntAttr(data=ub)), 
                                        "step": IntegerAttr(value=IntAttr(data=step))},
                            region=IRegion(block=IBlock(args=[IBlockArg() as iv], ops=ops))):   
                    new_ops: List[IOp] = []
                    # create an environment as values have to be remapped 
                    # i.e. use of IV has to be updated
                    env: Dict[ISSAValue, ISSAValue] = {}
                    for constant_iv in range(lb, ub, step):
                        iv_replacement = new_op(Constant,
                                attributes={
                                    "value": IntegerAttr.from_index_int_value(constant_iv)
                                }, result_types=[IndexType()])[-1]
                        assert iv_replacement.result is not None
                        new_ops.append(iv_replacement)
                        env |= {iv:iv_replacement.result}
                        for op_idx in range(0, len(ops)-1):
                            # As we supply an env, the operands will be updated accordingly. 
                            # The results of `ops[op_idx]` will be mapped to the results of this new op
                            # This is required to persist all uses between ops inside the former loop region
                            new_ops.extend(from_op(ops[op_idx], env=env))

                    result = new_ops

                    return success(result)
                case _:
                    return failure(self)


    @dataclass(frozen=True)
    class LoopSplit(Strategy):
        length_snd_loop: int

        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(op_type=affine.For, results=[IResult(typ=IntegerType() as type)], 
                            attributes={"lower_bound": IntegerAttr(value=IntAttr(data=lb)), 
                                        "upper_bound": IntegerAttr(value=IntAttr(data=ub)), 
                                        "step": IntegerAttr(value=IntAttr(data=step))},
                            region=IRegion(block=IBlock()) as loop_region) if ub > self.length_snd_loop:
                    assert step == 1 and "Splitting currently only supported for loops with step 1"

                    first_loop = from_op(op, attributes={"lower_bound": IntegerAttr.from_index_int_value(lb), 
                                                                "upper_bound": IntegerAttr.from_index_int_value(ub-self.length_snd_loop), 
                                                                "step": IntegerAttr.from_index_int_value(step)}, regions=[loop_region])
                    snd_loop = from_op(op, attributes={"lower_bound": IntegerAttr.from_index_int_value(ub-self.length_snd_loop), 
                                                                "upper_bound": IntegerAttr.from_index_int_value(ub), 
                                                                "step": IntegerAttr.from_index_int_value(step)}, regions=[loop_region])

                    return success(first_loop + snd_loop)
                case _:
                    return failure(self)


    ctx = MLContext()
    builtin.Builtin(ctx)
    func.Func(ctx)
    arith.Arith(ctx)
    scf.Scf(ctx)
    affine.Affine(ctx)
    memref.MemRef(ctx)

    parser = Parser(ctx, block_args_before)
    beforeM: Operation = parser.parse_op()
    immBeforeM: IOp = get_immutable_copy(beforeM)

    print("before:")
    printer = Printer()
    printer.print_op(beforeM)
    # test = topdown(seq(debug(), fail())).apply(immBeforeM)

    print("mutable_copy:")
    printer = Printer()
    printer.print_op(immBeforeM.get_mutable_copy())

    @dataclass(frozen=True)
    class First_nested_op(Strategy):
        s: Strategy

        def impl(self, op: IOp) -> RewriteResult:
            return region(block(elevate.op(self.s))).apply(op)
        
    @dataclass(frozen=True)
    class idAdd(Strategy):

        def impl(self, op: IOp) -> RewriteResult:
          if op.op_type == arith.Addi:
            return success(op)
          return failure(self)

    # rrImmM1 = First_nested_op(debug() ^ region(block(opsTopToBottom(debug() ^ LoopSplit(3))))).apply(immBeforeM)
    
    # rrImmM1 = backwards(idAdd()).apply(immBeforeM)

    # rrImmM1 = backwards(debug() ^ LoopSplit(3)).apply(immBeforeM)

    # rrImmM1 = backwards(debug() ^ LoopUnroll()).apply(rrImmM1.result_op)

    # rrImmM1 = region(block(opsTopToBottom(region(block(opsTopToBottom(CommuteAdd())))))).apply(immBeforeM)

    rrImmM1 = bottomToTop(debug() ^ CommuteAdd()).apply(immBeforeM)

    # rrImmM1 = region(block(opsTopToBottom(region(block(opsTopToBottom((id() ^ LoopSplit(3)))))))).apply(immBeforeM)
    
    print(rrImmM1)
    
    printer = Printer()
    printer.print_op(rrImmM1.result_op.get_mutable_copy())

    # rrImmM2 = region(block(opsTopToBottom(region(block(opsTopToBottom(debug() ^ LoopUnroll(), skips=1)))))).apply(rrImmM1.result_op)



    # rrImmM2 = skip(backwards() ,isa(affine.For)) ^ LoopUnroll().apply(rrImmM1.result_op)
    # printer = Printer()
    # printer.print_op(rrImmM2.result_op.get_mutable_copy())

    # rrImmM1 = topdown(debug() ^ fail()).apply(immBeforeM)
    # rrImmM1 = operand_rec(LoopSplit(3)).apply(immBeforeM)
    # print(rrImmM1)
    # assert rrImmM1.isSuccess()

    # printer = Printer()
    # printer.print_op(rrImmM1.result_op.get_mutable_copy())

    # rrImmM2 = operand_rec(debug() ^ skip(isa(affine.For)) ^ LoopUnroll()).apply(rrImmM1.result_op)
    # print(rrImmM2)
    # assert (rrImmM2.isSuccess()
    #         and isinstance(rrImmM2.result_op, IOp))

    # file = StringIO("")
    # printer = Printer(stream=file)
    # printer.print_op(rrImmM2.result_op.get_mutable_copy())

    # For debugging: printing the actual output
    # print("after:")
    # print(file.getvalue().strip())

    # checkDiff = False
    # if checkDiff:
    #     diff = difflib.Differ().compare(file.getvalue().splitlines(True),
    #                                     expected.splitlines(True))
    #     print(''.join(diff))
    #     assert file.getvalue().strip() == expected.strip()


if __name__ == "__main__":
    rewriting_with_immutability_experiments()
