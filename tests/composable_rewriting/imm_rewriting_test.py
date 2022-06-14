from __future__ import annotations
from io import StringIO
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.func import *
from xdsl.elevate import *
from xdsl.immutable_ir import *
from xdsl.immutable_utils import *
import difflib


def apply_strategy_and_compare(program: str, expected_program: str,
                               strategy: Strategy):
    ctx = MLContext()
    Builtin(ctx)
    Func(ctx)
    Arith(ctx)
    scf.Scf(ctx)

    parser = Parser(ctx, program)
    module: Operation = parser.parse_op()
    imm_module: IOp = get_immutable_copy(module)

    rr = strategy.apply(imm_module)
    assert rr.isSuccess()

    # for debugging
    printer = Printer()
    print(f'Result after applying "{strategy}":')
    printer.print_op(rr.result_op.get_mutable_copy())
    print()

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(rr.result_op.get_mutable_copy())

    diff = difflib.Differ().compare(file.getvalue().splitlines(True),
                                    expected_program.splitlines(True))
    if file.getvalue().strip() != expected_program.strip():
        print("Did not get expected output! Diff:")
        print(''.join(diff))
        assert False


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
class FoldConstantAdd(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        match op:
          case IOp(op_type=arith.Addi,
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
class ChangeConstantTo42(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        match op:
          case IOp(op_type=arith.Constant, 
                    attributes={"value": IntegerAttr(typ=IntegerType(width=IntAttr(data=32))) as attr}):
              result = from_op(op,
                        attributes={
                            "value":
                            IntegerAttr.from_params(42, attr.typ)
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
                        region=IRegion(ops=ops)):
                        return success(ops[:-1] if len(ops) > 0 and (ops[-1].op_type==scf.Yield) else ops)
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
class RemoveAddZero(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        match op:
            case IOp(op_type=arith.Addi,
                    operands=[ISSAValue(op=IOp(op_type=arith.Constant, 
                                                  attributes={"value": IntegerAttr(value=IntAttr(data=0))})), 
                                    IResult() as operand2]):
                id = new_op(RewriteId, operands=[operand2], result_types=[operand2.typ])
                return success(id)
            case _:
                return failure(self)

@dataclass(frozen=True)
class Mul2ToShift(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        match op:
            case IOp(op_type=arith.Muli, results=[IResult(typ=IntegerType() as type)], 
                      operands=[ISSAValue() as input, IResult(op=IOp(op_type=arith.Constant, attributes={"value": IntegerAttr(value=IntAttr(data=2))}))]):
                return success(new_op(arith.ShLI, 
                                        result_types=[type], 
                                        operands=[input, 
                                                  new_op(Constant,
                                                      attributes={
                                                          "value": IntegerAttr.from_params(1, type)
                                                      }, result_types=[type])]
                            ))
            case _:
                return failure(self)

def test_double_commute():
    """Tests a strategy which swaps the two operands of an arith.addi."""

    before = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 4 : !i32]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  %2 : !i32 = arith.constant() ["value" = 1 : !i32]
  %3 : !i32 = arith.addi(%2 : !i32, %1 : !i32)
  %4 : !i32 = arith.addi(%3 : !i32, %0 : !i32)
  func.return(%4 : !i32)
}
"""
    once_commuted = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 4 : !i32]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  %2 : !i32 = arith.constant() ["value" = 1 : !i32]
  %3 : !i32 = arith.addi(%2 : !i32, %1 : !i32)
  %4 : !i32 = arith.addi(%0 : !i32, %3 : !i32)
  func.return(%4 : !i32)
}
"""
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=before,
                               expected_program=once_commuted,
                               strategy=backwards(CommuteAdd()))
    apply_strategy_and_compare(program=before,
                               expected_program=once_commuted,
                               strategy=topToBottom(CommuteAdd(), skips=1))
    apply_strategy_and_compare(program=before,
                               expected_program=once_commuted,
                               strategy=bottomToTop(CommuteAdd()))
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=once_commuted,
                               expected_program=before,
                               strategy=backwards(CommuteAdd()))    
    apply_strategy_and_compare(program=once_commuted,
                               expected_program=before,
                               strategy=topToBottom(CommuteAdd(), skips=1))
    apply_strategy_and_compare(program=once_commuted,
                               expected_program=before,
                               strategy=bottomToTop(CommuteAdd()))
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=before,
                               expected_program=before,
                               strategy=seq(backwards(CommuteAdd()),
                                            backwards(CommuteAdd())))
    apply_strategy_and_compare(program=before,
                               expected_program=before,
                               strategy=seq(topToBottom(CommuteAdd()),
                                            topToBottom(CommuteAdd())))
    apply_strategy_and_compare(program=before,
                               expected_program=before,
                               strategy=seq(bottomToTop(CommuteAdd()),
                                            bottomToTop(CommuteAdd())))

def test_commute_block_args():
    """Tests a strategy which swaps the two operands of 
    an arith.addi which are block_args."""

    before = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
    func.return(%2 : !i32)
  }
}
"""
    commuted = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i32 = arith.addi(%1 : !i32, %0 : !i32)
    func.return(%2 : !i32)
  }
}
"""
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=before,
                               expected_program=commuted,
                               strategy=backwards(CommuteAdd()))
    apply_strategy_and_compare(program=before,
                               expected_program=commuted,
                               strategy=topToBottom(CommuteAdd()))
    apply_strategy_and_compare(program=before,
                               expected_program=commuted,
                               strategy=bottomToTop(CommuteAdd()))
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=commuted,
                               expected_program=before,
                               strategy=backwards(CommuteAdd()))
    apply_strategy_and_compare(program=commuted,
                               expected_program=before,
                               strategy=topToBottom(CommuteAdd()))
    apply_strategy_and_compare(program=commuted,
                               expected_program=before,
                               strategy=bottomToTop(CommuteAdd()))
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=before,
                               expected_program=before,
                               strategy=seq(
                                        backwards(CommuteAdd()),
                                        backwards(CommuteAdd())))
    apply_strategy_and_compare(program=before,
                               expected_program=before,
                               strategy=seq(
                                        topToBottom(CommuteAdd()),
                                        topToBottom(CommuteAdd())))
    apply_strategy_and_compare(program=before,
                               expected_program=before,
                               strategy=seq(
                                        bottomToTop(CommuteAdd()),
                                        bottomToTop(CommuteAdd())))


def test_rewriting_with_blocks():
    before = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i1):
    %1 : !i32 = scf.if(%0 : !i1) {
      %2 : !i32 = arith.constant() ["value" = 0 : !i32]
      scf.yield(%2 : !i32)
    }
    func.return(%1 : !i32)
  }
}
"""
    constant_changed = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i1):
    %1 : !i32 = scf.if(%0 : !i1) {
      %2 : !i32 = arith.constant() ["value" = 42 : !i32]
      scf.yield(%2 : !i32)
    }
    func.return(%1 : !i32)
  }
}
"""
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=before,
                               expected_program=constant_changed,
                               strategy=backwards(ChangeConstantTo42()))
    apply_strategy_and_compare(program=before,
                               expected_program=constant_changed,
                               strategy=topToBottom(ChangeConstantTo42()))
    apply_strategy_and_compare(program=before,
                               expected_program=constant_changed,
                               strategy=bottomToTop(ChangeConstantTo42()))

def test_constant_folding():
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
    once_folded = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  %2 : !i32 = arith.constant() ["value" = 3 : !i32]
  %3 : !i32 = arith.constant() ["value" = 4 : !i32]
  %4 : !i32 = arith.addi(%2 : !i32, %3 : !i32)
  func.return(%4 : !i32)
}
"""
    twice_folded = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  %2 : !i32 = arith.constant() ["value" = 3 : !i32]
  %3 : !i32 = arith.constant() ["value" = 4 : !i32]
  %4 : !i32 = arith.constant() ["value" = 7 : !i32]
  func.return(%4 : !i32)
}
"""
    twice_folded_garbage_collected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 7 : !i32]
  func.return(%0 : !i32)
}
"""
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=before,
                               expected_program=once_folded,
                               strategy=backwards(FoldConstantAdd()))
    apply_strategy_and_compare(program=before,
                               expected_program=once_folded,
                               strategy=topToBottom(FoldConstantAdd()))
    apply_strategy_and_compare(program=before,
                               expected_program=once_folded,
                               strategy=(bottomToTop(FoldConstantAdd())))
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=once_folded,
                               expected_program=twice_folded,
                               strategy=backwards(FoldConstantAdd()))
    apply_strategy_and_compare(program=once_folded,
                               expected_program=twice_folded,
                               strategy=topToBottom(FoldConstantAdd()))
    apply_strategy_and_compare(program=once_folded,
                               expected_program=twice_folded,
                               strategy=bottomToTop(FoldConstantAdd()))
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=before,
                               expected_program=twice_folded,
                               strategy=seq(backwards(FoldConstantAdd()),
                                            backwards(FoldConstantAdd())))
    apply_strategy_and_compare(program=before,
                               expected_program=twice_folded,
                               strategy=seq(topToBottom(FoldConstantAdd()),
                                            topToBottom(FoldConstantAdd())))
    apply_strategy_and_compare(program=before,
                               expected_program=twice_folded,
                               strategy=seq(bottomToTop(FoldConstantAdd()),
                                            bottomToTop(FoldConstantAdd())))
    # Garbage Collection
    apply_strategy_and_compare(program=twice_folded,
                               expected_program=twice_folded_garbage_collected,
                               strategy=GarbageCollect())
def test_inline_if():
    before = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[], [!i32]>, "sym_visibility" = "private"] {
  ^0():
    %0 : !i1 = arith.constant() ["value" = 1 : !i1]
    %1 : !i32 = scf.if(%0 : !i1) {
      %2 : !i32 = arith.constant() ["value" = 42 : !i32]
      scf.yield(%2 : !i32)
    }
    func.return(%1 : !i32)
  }
}
"""
    inlined = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[], [!i32]>, "sym_visibility" = "private"] {
    %0 : !i1 = arith.constant() ["value" = 1 : !i1]
    %1 : !i32 = arith.constant() ["value" = 42 : !i32]
    func.return(%1 : !i32)
  }
}
"""
    inlined_garbage_collected = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[], [!i32]>, "sym_visibility" = "private"] {
    %0 : !i32 = arith.constant() ["value" = 42 : !i32]
    func.return(%0 : !i32)
  }
}
"""
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=before,
                               expected_program=inlined,
                               strategy=backwards(InlineIf()))
    apply_strategy_and_compare(program=before,
                               expected_program=inlined,
                               strategy=topToBottom(InlineIf()))
    apply_strategy_and_compare(program=before,
                               expected_program=inlined,
                               strategy=bottomToTop(InlineIf()))
    # Garbage Collection
    apply_strategy_and_compare(program=inlined,
                               expected_program=inlined_garbage_collected,
                               strategy=GarbageCollect())

def test_inline_and_fold():
    before = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[], [!i32]>, "sym_visibility" = "private"] {
  ^0():
    %0 : !i1 = arith.constant() ["value" = 1 : !i1]
    %1 : !i32 = scf.if(%0 : !i1) {
      %2 : !i32 = arith.constant() ["value" = 1 : !i32]
      %3 : !i32 = arith.constant() ["value" = 2 : !i32]
      %4 : !i32 = arith.addi(%2 : !i32, %3 : !i32)
      %5 : !i32 = arith.constant() ["value" = 4 : !i32]
      %6 : !i32 = arith.addi(%4 : !i32, %5 : !i32)
      scf.yield(%6 : !i32)
    }
    func.return(%1 : !i32)
  }
}
"""

    inlined = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[], [!i32]>, "sym_visibility" = "private"] {
    %0 : !i1 = arith.constant() ["value" = 1 : !i1]
    %1 : !i32 = arith.constant() ["value" = 1 : !i32]
    %2 : !i32 = arith.constant() ["value" = 2 : !i32]
    %3 : !i32 = arith.addi(%1 : !i32, %2 : !i32)
    %4 : !i32 = arith.constant() ["value" = 4 : !i32]
    %5 : !i32 = arith.addi(%3 : !i32, %4 : !i32)
    func.return(%5 : !i32)
  }
}
"""
    folded_and_inlined = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[], [!i32]>, "sym_visibility" = "private"] {
    %0 : !i1 = arith.constant() ["value" = 1 : !i1]
    %1 : !i32 = arith.constant() ["value" = 1 : !i32]
    %2 : !i32 = arith.constant() ["value" = 2 : !i32]
    %3 : !i32 = arith.constant() ["value" = 3 : !i32]
    %4 : !i32 = arith.constant() ["value" = 4 : !i32]
    %5 : !i32 = arith.constant() ["value" = 7 : !i32]
    func.return(%5 : !i32)
  }
}
"""
    folded_and_inlined_garbage_collected = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[], [!i32]>, "sym_visibility" = "private"] {
    %0 : !i32 = arith.constant() ["value" = 7 : !i32]
    func.return(%0 : !i32)
  }
}
"""
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=before,
                               expected_program=inlined,
                               strategy=backwards(InlineIf()))
    apply_strategy_and_compare(program=before,
                               expected_program=inlined,
                               strategy=topToBottom(InlineIf()))
    apply_strategy_and_compare(program=before,
                               expected_program=inlined,
                               strategy=bottomToTop(InlineIf()))
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=before,
                               expected_program=folded_and_inlined,
                               strategy=seq(backwards(InlineIf()), 
                                            seq(backwards(FoldConstantAdd()), 
                                                backwards(FoldConstantAdd()))))
    apply_strategy_and_compare(program=before,
                               expected_program=folded_and_inlined,
                               strategy=seq(topToBottom(InlineIf()), 
                                            seq(topToBottom(FoldConstantAdd()), 
                                                topToBottom(FoldConstantAdd()))))
    apply_strategy_and_compare(program=before,
                               expected_program=folded_and_inlined,
                               strategy=seq(bottomToTop(InlineIf()), 
                                            seq(bottomToTop(FoldConstantAdd()), 
                                                bottomToTop(FoldConstantAdd()))))
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=before,
                               expected_program=folded_and_inlined,
                               strategy=seq(backwards(FoldConstantAdd()), 
                                            seq(backwards(FoldConstantAdd()), 
                                                backwards(InlineIf()))))
    apply_strategy_and_compare(program=before,
                               expected_program=folded_and_inlined,
                               strategy=seq(topToBottom(FoldConstantAdd()), 
                                            seq(topToBottom(FoldConstantAdd()), 
                                                topToBottom(InlineIf()))))
    apply_strategy_and_compare(program=before,
                               expected_program=folded_and_inlined,
                               strategy=seq(bottomToTop(FoldConstantAdd()), 
                                            seq(bottomToTop(FoldConstantAdd()), 
                                                bottomToTop(InlineIf()))))
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=before,
                               expected_program=folded_and_inlined,
                               strategy=seq(backwards(FoldConstantAdd()), 
                                            seq(backwards(InlineIf()), 
                                                backwards(FoldConstantAdd()))))
    apply_strategy_and_compare(program=before,
                               expected_program=folded_and_inlined,
                               strategy=seq(topToBottom(FoldConstantAdd()), 
                                            seq(topToBottom(InlineIf()), 
                                                topToBottom(FoldConstantAdd()))))
    apply_strategy_and_compare(program=before,
                               expected_program=folded_and_inlined,
                               strategy=seq(bottomToTop(FoldConstantAdd()), 
                                            seq(bottomToTop(InlineIf()), 
                                                bottomToTop(FoldConstantAdd()))))
    # Garbage Collection             
    apply_strategy_and_compare(program=folded_and_inlined,
                               expected_program=folded_and_inlined_garbage_collected,
                               strategy=GarbageCollect())
def test_add_zero():
    before = \
"""module() {
%0 : !i32 = arith.constant() ["value" = 1 : !i32]
func.return(%0 : !i32)
}
"""
    added_zero = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
  %1 : !i32 = arith.constant() ["value" = 0 : !i32]
  %2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
  func.return(%2 : !i32)
}
"""
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=before,
                              expected_program=added_zero,
                              strategy=backwards(AddZero()))
    apply_strategy_and_compare(program=before,
                              expected_program=added_zero,
                              strategy=topToBottom(AddZero()))
    apply_strategy_and_compare(program=before,
                              expected_program=added_zero,
                              strategy=bottomToTop(AddZero()))

def test_remove_add_zero():
    before = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
  %1 : !i32 = arith.constant() ["value" = 0 : !i32]
  %2 : !i32 = arith.addi(%1 : !i32, %0 : !i32)
  func.return(%2 : !i32)
}
"""
    removed_add_zero = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
  %1 : !i32 = arith.constant() ["value" = 0 : !i32]
  func.return(%0 : !i32)
}
"""
    removed_add_zero_garbage_collected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
  func.return(%0 : !i32)
}
"""
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=before,
                              expected_program=removed_add_zero,
                              strategy=backwards(RemoveAddZero()))
    apply_strategy_and_compare(program=before,
                              expected_program=removed_add_zero,
                              strategy=topToBottom(RemoveAddZero()))
    apply_strategy_and_compare(program=before,
                              expected_program=removed_add_zero,
                              strategy=bottomToTop(RemoveAddZero()))
    # Garbage Collection
    apply_strategy_and_compare(program=removed_add_zero,
                              expected_program=removed_add_zero_garbage_collected,
                              strategy=bottomToTop(GarbageCollect()))

def test_deeper_nested_block_args_commute():
    """This test demonstrates that substitution of block arguments also works when the 
    concerned op is at a deeper nesting level than the block supplying the block args.
    """
  
    before = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i1 = arith.constant() ["value" = 1 : !i1]
    %3 : !i32 = scf.if(%2 : !i1) {
      %4 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
      scf.yield(%4 : !i32)
    }
    func.return(%3 : !i32)
  }
}
"""
    nested_commute = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i1 = arith.constant() ["value" = 1 : !i1]
    %3 : !i32 = scf.if(%2 : !i1) {
      %4 : !i32 = arith.addi(%1 : !i32, %0 : !i32)
      scf.yield(%4 : !i32)
    }
    func.return(%3 : !i32)
  }
}
"""
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=before,
                              expected_program=nested_commute,
                              strategy=backwards(CommuteAdd()))
    apply_strategy_and_compare(program=before,
                              expected_program=nested_commute,
                              strategy=topToBottom(CommuteAdd()))
    apply_strategy_and_compare(program=before,
                              expected_program=nested_commute,
                              strategy=bottomToTop(CommuteAdd()))

def test_mul2_to_lshift():
    """
    """
  
    before = \
"""module() {
  func.func() ["sym_name" = "times_2", "type" = !fun<[!i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32):
    %1 : !i32 = arith.constant() ["value" = 2 : !i32]
    %2 : !i32 = arith.muli(%0 : !i32, %1 : !i32)
    func.return(%2 : !i32)
  }
}
"""
    left_shifted = \
"""module() {
  func.func() ["sym_name" = "times_2", "type" = !fun<[!i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32):
    %1 : !i32 = arith.constant() ["value" = 2 : !i32]
    %2 : !i32 = arith.constant() ["value" = 1 : !i32]
    %3 : !i32 = arith.shli(%0 : !i32, %2 : !i32)
    func.return(%3 : !i32)
  }
}
"""
    left_shifted_garbage_collected = \
"""module() {
  func.func() ["sym_name" = "times_2", "type" = !fun<[!i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32):
    %1 : !i32 = arith.constant() ["value" = 1 : !i32]
    %2 : !i32 = arith.shli(%0 : !i32, %1 : !i32)
    func.return(%2 : !i32)
  }
}
"""
    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=before,
                              expected_program=left_shifted,
                              strategy=backwards(Mul2ToShift()))
    apply_strategy_and_compare(program=before,
                              expected_program=left_shifted,
                              strategy=topToBottom(Mul2ToShift()))
    # Garbage Collection
    apply_strategy_and_compare(program=left_shifted,
                              expected_program=left_shifted_garbage_collected,
                              strategy=GarbageCollect())

                          
if __name__ == "__main__":
    test_double_commute()
    test_commute_block_args()
    test_rewriting_with_blocks()
    # TODO: rewriting with successors
    test_constant_folding()
    test_inline_if()
    test_inline_and_fold()
    test_add_zero()
    test_remove_add_zero()
    test_deeper_nested_block_args_commute()
    test_mul2_to_lshift()