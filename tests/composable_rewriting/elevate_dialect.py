from __future__ import annotations
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
import xdsl.dialects.func as func
import xdsl.dialects.builtin as builtin
import xdsl.dialects.match.dialect as match
import xdsl.dialects.rewrite.dialect as rewrite
import xdsl.dialects.elevate.dialect as elevate
import xdsl.dialects.elevate.interpreter as interpreter

from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.func import *
from xdsl.elevate import *
from xdsl.immutable_ir import *
from xdsl.immutable_utils import *


def apply_strategy(program: str, strategy: Strategy):
    ctx = MLContext()
    Builtin(ctx)
    Func(ctx)
    Arith(ctx)
    scf.Scf(ctx)
    match.Match(ctx)
    rewrite.Rewrite(ctx)

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


def parse_module(program: str) -> ModuleOp:
    ctx = MLContext()
    Builtin(ctx)
    Func(ctx)
    Arith(ctx)
    scf.Scf(ctx)
    elevate.Elevate(ctx)
    match.Match(ctx)
    rewrite.Rewrite(ctx)

    parser = Parser(ctx, program)
    module: Operation = parser.parse_op()

    printer = Printer()
    printer.print_op(module)

    assert isinstance(module, ModuleOp)
    return module


def interpret_elevate_2(strategy: elevate.StrategyStratOp) -> Strategy:
    # Build strategy:
    returnOp = strategy.body.ops[-1]
    assert isinstance(returnOp, elevate.ReturnStratOp)

    def get_strategy(op: elevate.ElevateOperation):
        strat: Type[Strategy] = op.get_strategy()
        operands_to_strat: List[Strategy] = [
            get_strategy(operand.op) for operand in op.operands
        ]

        return strat(*operands_to_strat)

    return get_strategy(returnOp.operands[0].op)


def test_region_based_dialect():
    # In this case combinators as operations with regions

    ir_to_rewrite = \
"""module() {
%0 : !i32 = arith.constant() ["value" = 1 : !i32]
%1 : !i32 = arith.constant() ["value" = 2 : !i32]
%2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
%3 : !i32 = arith.constant() ["value" = 4 : !i32]
%4 : !i32 = arith.addi(%2 : !i32, %3 : !i32)
func.return(%4 : !i32)
}
"""

    constant_fold_strat = \
"""module() {
  %0 : !strategy = elevate.strategy() {
    %int_type : !type<!i32> = match.type()

    // match the first constant
    %attr1 : !i32 = match.attr() ["name" = "value"]
    %3 : !operation = match.op(%attr1 : !i32, %int_type : !type<!i32>) ["name"="arith.constant"]
    %4 : !value = match.get_result(%3 : !operation) ["idx" = 0]

    // match the second constant
    %attr2 : !i32 = match.attr() ["name" = "value"]
    %5 : !operation = match.op(%attr2 : !i32, %int_type : !type<!i32>) ["name"="arith.constant"]
    %6 : !value = match.get_result(%5 : !operation) ["idx" = 0]

    // match the addition of these constants
    %7 : !operation = match.root_op(%4 : !value, %6 : !value, %int_type : !type<!i32>) ["name"="arith.addi"]

    // rewriting
    %new_attr_val : !i32 = arith.addi(%attr1 : !i32, %attr2 : !i32)
    %result : !operation = rewrite.new_op(%new_attr_val : !i32, %int_type : !type<!i32>) ["name" = "arith.constant", "attribute_names"=["value"]]
    rewrite.success(%result : !operation)
  }
  %100 : !strategy = elevate.compose() {
    elevate.toptobottom() {
      //elevate.try() {
            elevate.apply(%0 : !strategy)
      //}
      elevate.id()
    }
  }
}
"""

    composed_strat_dsl_actually_python_question = \
"""module() {
  %0 : !strategy = elevate.strategy() {
    match()

  } { 
    // Here we have the rewriting portion

  }
  %1 : !strategy = elevate.compose() {
    elevate.toptobottom() {
      elevate.try() {
            elevate.apply(%0 : !strategy)
      }
      elevate.id()
    }
  }
}
"""

    module = parse_module(constant_fold_strat)
    elevate_interpreter = interpreter.ElevateInterpreter()
    print(strategy := elevate_interpreter.get_strategy(module))

    apply_strategy(ir_to_rewrite, strategy)
    apply_strategy(ir_to_rewrite, strategy ^ strategy)


def test_strats_explicit_apply():
    # This mimics how we use elevate right now much more closely.

    ir = \
"""module() {
  %0 : !strategy_handle = elevate2.strategy() {
  ^0(%1 : !op_handle):
    %2 : !strategy_handle = elevate2.fail()
    %3 : !strategy_handle = elevate2.try(%2 : !strategy_handle)
    %4 : !strategy_handle = elevate2.id()
    %5 : !strategy_handle = elevate2.seq(%3 : !strategy_handle, %4 : !strategy_handle)
    %6 : !strategy_handle = elevate2.toptobottom(%5 : !strategy_handle)

    elevate2.return(%6 : !strategy_handle)
  }
}
"""
    module = parse_module(ir)
    assert isinstance(module.ops[0], elevate.StrategyStratOp)
    print(interpret_elevate_2(module.ops[0]))
    # prints: topToBottom(seq(try_(fail()),id()),0)


if __name__ == "__main__":
    # test_strats_explicit_apply()
    test_region_based_dialect()