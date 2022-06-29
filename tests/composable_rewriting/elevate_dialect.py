from __future__ import annotations
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
import xdsl.dialects.func as func
import xdsl.dialects.builtin as builtin
import xdsl.dialects.elevate_dialect as elevate
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.func import *
from xdsl.elevate import *
from xdsl.immutable_ir import *
from xdsl.immutable_utils import *


def parse_module(program: str) -> ModuleOp:
    ctx = MLContext()
    Builtin(ctx)
    Func(ctx)
    Arith(ctx)
    scf.Scf(ctx)
    elevate.Elevate(ctx)

    parser = Parser(ctx, program)
    module: Operation = parser.parse_op()

    printer = Printer()
    printer.print_op(module)

    assert isinstance(module, ModuleOp)
    return module


def interpret_elevate(strategy: elevate.StrategyOp) -> Strategy:
    # Build strategy:

    def get_strategy_for_region(region: Region) -> Strategy:
        assert len(region.ops) > 0
        strategy = get_strategy(region.ops[0])
        for idx in range(1, len(region.ops)):
            # TODO: assertion that all ops in the region are ElevateOps
            if isinstance(region.ops[idx], elevate.ReturnOp):
                break
            strategy = seq(strategy, get_strategy(region.ops[idx]))
        return strategy

    def get_strategy(op: elevate.ElevateOperation) -> Strategy:
        if len(op.regions) == 0:
            return op.get_strategy()()
        
        strategy_type: Type[Strategy] = op.__class__.get_strategy()
        operands_to_strat: List[Strategy] = [get_strategy_for_region(region) for region in op.regions]
    
        return strategy_type(*operands_to_strat)

    return get_strategy_for_region(strategy.body)
    

def interpret_elevate_2(strategy: elevate.StrategyStratOp) -> Strategy:
    # Build strategy:
    returnOp = strategy.body.ops[-1]
    assert isinstance(returnOp, elevate.ReturnStratOp)

    def get_strategy(op: elevate.ElevateOperation):
        strat: Type[Strategy] = op.get_strategy()
        operands_to_strat: List[Strategy] = [get_strategy(operand.op) for operand in op.operands]
    
        return strat(*operands_to_strat)

    return get_strategy(returnOp.operands[0].op)


def test_strats_produce_ophandles():
    # In this case combinators as operations with regions


    # The types here are not really proper. 
    # I think everything should probably become !strategy_handle
    ir = \
"""module() {
  %0 : !op_handle = elevate.strategy() {
    ^0(%1 : !op_handle):
    %2 : !op_handle = elevate.toptobottom(%1 : !op_handle) {
      ^1(%3 : !op_handle):
      %4 : !op_handle = elevate.try(%3 : !op_handle) {
        ^3(%5 : !op_handle):
        %6 : !op_handle = elevate.fail(%5 : !op_handle)
      }
      %7 : !op_handle = elevate.id(%4 : !op_handle)
      elevate.return(%7 : !op_handle)
    }
    elevate.return(%2 : !op_handle)
  }
}
"""
    module = parse_module(ir)
    assert isinstance(module.ops[0], elevate.StrategyOp)
    print(interpret_elevate(module.ops[0]))
    # prints: topToBottom(seq(try_(fail()),id()),0)

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
    test_strats_produce_ophandles()
    test_strats_explicit_apply()