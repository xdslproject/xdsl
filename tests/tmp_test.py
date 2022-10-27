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
import os
import xdsl.dialects.IRUtils.dialect as IRUtils
import xdsl.dialects.pdl.dialect as pdl
import xdsl.dialects.match.dialect as match
import xdsl.dialects.rewrite.dialect as rewrite
import xdsl.dialects.elevate.dialect as elevate
import xdsl.dialects.elevate.interpreter as interpreter
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
        # print(''.join(diff))
        assert False


def apply_dyn_strategy_and_compare(program: str, expected_program: str,
                               strategy_name: str):
    ctx = MLContext()
    Builtin(ctx)
    Func(ctx)
    Arith(ctx)
    scf.Scf(ctx)
    IRUtils.IRUtils(ctx)
    pdl.PDL(ctx)
    match.Match(ctx)
    rewrite.Rewrite(ctx)
    elevate.Elevate(ctx)

    # fetch strategies.xdsl file
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    strategy_file = open(os.path.join(__location__, 'strategies.xdsl'))
    strategy_string = strategy_file.read()

    ir_parser = Parser(ctx, program)
    ir_module: Operation = ir_parser.parse_op()
    imm_ir_module: IOp = get_immutable_copy(ir_module)

    strat_parser = Parser(ctx, strategy_string)
    strat_module: Operation = strat_parser.parse_op()
    elevate_interpreter = interpreter.ElevateInterpreter()
    elevate_interpreter.register_native_strategy(GarbageCollect,
                                                 "garbage_collect")

    strategies = elevate_interpreter.get_strategy(strat_module)
    strategy = strategies[strategy_name]

    rr = strategy.apply(imm_ir_module)
    assert rr.isSuccess()

    # for debugging
    printer = Printer()
    print(f'Result after applying "{strategy}":')
    printer.print_op(rr.result_op.get_mutable_copy())

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
class FoldConstantAdd(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        match op:
            case IOp(
                op_type=arith.Addi,
                operands=[IResult(op=IOp(op_type=arith.Constant,
                                        attributes={"value": IntegerAttr() as attr1}) as c1),
                          IResult(op=IOp(op_type=arith.Constant,
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
class FakeDCE(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        match op:
            case IOp(op_type=ModuleOp):
                new_region = IRegion([IBlock([], op.region.ops[2:])])
                result = from_op(op, regions=[new_region])
                return success(result)
            case _:
                return failure(self)

def test_double_commute():
    """Tests a strategy which swaps the two operands of an arith.addi."""

    before = \
"""builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
  %1 : !i32 = arith.constant() ["value" = 1 : !i32]
  %2 : !i32 = arith.addi(%1 : !i32, %0 : !i32)
  %3 : !i32 = arith.constant() ["value" = 1 : !i32]
  %4 : !i32 = arith.addi(%3 : !i32, %2 : !i32)
  %5 : !i32 = arith.constant() ["value" = 1 : !i32]
  %6 : !i32 = arith.addi(%5 : !i32, %4 : !i32)
  %7 : !i32 = arith.constant() ["value" = 1 : !i32]
  %8 : !i32 = arith.addi(%7 : !i32, %6 : !i32)
  %9 : !i32 = arith.constant() ["value" = 1 : !i32]
  %10 : !i32 = arith.addi(%9 : !i32, %8 : !i32)
  %11 : !i32 = arith.constant() ["value" = 1 : !i32]
  %12 : !i32 = arith.addi(%11 : !i32, %10 : !i32)
  %13 : !i32 = arith.constant() ["value" = 1 : !i32]
  %14 : !i32 = arith.addi(%13 : !i32, %12 : !i32)
  %15 : !i32 = arith.constant() ["value" = 1 : !i32]
  %16 : !i32 = arith.addi(%15 : !i32, %14 : !i32)
  %17 : !i32 = arith.constant() ["value" = 1 : !i32]
  %18 : !i32 = arith.addi(%17 : !i32, %16 : !i32)
  %19 : !i32 = arith.constant() ["value" = 1 : !i32]
  %20 : !i32 = arith.addi(%19 : !i32, %18 : !i32)
  func.return(%20 : !i32)
}
"""
    once_commuted = \
"""builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 4 : !i32]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  %2 : !i32 = arith.constant() ["value" = 1 : !i32]
  %3 : !i32 = arith.addi(%2 : !i32, %1 : !i32)
  %4 : !i32 = arith.addi(%0 : !i32, %3 : !i32)
  func.return(%4 : !i32)
}
"""

    strategy = try_(topToBottom(FoldConstantAdd()))
    
    for _ in range(10):
        strategy =  strategy ^ (topToBottom(FakeDCE())) ^ try_(topToBottom(FoldConstantAdd())) 

    # similar rewrite, different traversals:
    apply_strategy_and_compare(program=before,
                               expected_program=once_commuted,
                               strategy=strategy) #everywhere(topToBottom(FoldConstantAdd()) ^ (FakeDCE())))

def test_bool_nest():
    @dataclass(frozen=True)
    class FoldConstantAnd(Strategy):
    
        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(
                    op_type=arith.AndI,
                    operands=[IResult(op=IOp(op_type=arith.Constant,
                                            attributes={"value": IntegerAttr() as attr1}) as c1),
                              IResult(op=IOp(op_type=arith.Constant,
                                            attributes={"value": IntegerAttr() as attr2}))]):
    
                    result = from_op(c1,
                            attributes={
                                "value":
                                IntegerAttr.from_params(
                                    attr1.value.data & attr2.value.data,
                                    attr1.typ)
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

    before = \
"""builtin.module() {
  func.func() ["sym_name" = "main", "function_type" = !fun<[], [!i1, !i1]>, "sym_visibility" = "private"] {
    %0 : !i1 = arith.constant() ["value" = true]
    %1 : !i1 = scf.if(%0 : !i1) {
      %2 : !i1 = arith.constant() ["value" = true]
      %3 : !i1 = arith.constant() ["value" = true]
      %4 : !i1 = arith.andi(%3 : !i1, %2 : !i1)
      %5 : !i1 = arith.constant() ["value" = true]
      %6 : !i1 = arith.andi(%5 : !i1, %4 : !i1)
      %7 : !i1 = arith.constant() ["value" = true]
      %8 : !i1 = arith.andi(%7 : !i1, %6 : !i1)
      %9 : !i1 = arith.constant() ["value" = true]
      %10 : !i1 = arith.andi(%9 : !i1, %8 : !i1)
      %11 : !i1 = arith.constant() ["value" = true]
      %12 : !i1 = arith.andi(%11 : !i1, %10 : !i1)
      scf.yield(%12 : !i1)
    } {
      %13 : !i1 = arith.constant() ["value" = false]
      scf.yield(%13 : !i1)
    }
    %14 : !i1 = scf.if(%0 : !i1) {
      %15 : !i1 = arith.constant() ["value" = true]
      %16 : !i1 = arith.constant() ["value" = true]
      %17 : !i1 = arith.andi(%16 : !i1, %15 : !i1)
      %18 : !i1 = arith.constant() ["value" = true]
      %19 : !i1 = arith.andi(%18 : !i1, %17 : !i1)
      %20 : !i1 = arith.constant() ["value" = true]
      %21 : !i1 = arith.andi(%20 : !i1, %19 : !i1)
      %22 : !i1 = arith.constant() ["value" = true]
      %23 : !i1 = arith.andi(%22 : !i1, %21 : !i1)
      %24 : !i1 = arith.constant() ["value" = true]
      %25 : !i1 = arith.andi(%24 : !i1, %23 : !i1)
      scf.yield(%25 : !i1)
    } {
      %26 : !i1 = arith.constant() ["value" = false]
      scf.yield(%26 : !i1)
    }
    func.return(%1 : !i1, %14 : !i1)
  }
}
"""
    apply_strategy_and_compare(program=before,
                                expected_program=before,
                                strategy=everywhere(FoldConstantAnd())  ^ everywhere(InlineIf()))
if __name__ == "__main__":
    # test_double_commute()
    test_bool_nest()