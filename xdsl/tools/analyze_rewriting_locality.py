from math import ceil
from xdsl.dialects.elevate.dialect import ReturnOp
from xdsl.ir import MLContext, OpResult
from xdsl.dialects.arith import Addi, Arith, Constant, AndI
from xdsl.dialects.func import *
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
import xdsl.dialects.onnx.dialect as onnx
import xdsl.dialects.stencil.stencil as stencil
from xdsl.dialects.builtin import Builtin, IntAttr, IntegerAttr, IntegerType, ModuleOp
from xdsl.parser import Parser
from xdsl.printer import Printer
import argparse
from sys import stdin
import statistics

test_prog: str = """
builtin.module() {
  func.func() ["sym_name" = "main", "function_type" = !fun<[], [!i1, !i1]>, "sym_visibility" = "private"] {
    %0 : !i1 = arith.constant() ["value" = true]
    %1 : !i1 = scf.if(%0 : !i1) {
      %2 : !i1 = arith.constant() ["value" = true]
      %3 : !i1 = arith.constant() ["value" = true]
      %4 : !i1 = arith.andi(%3 : !i1, %2 : !i1)
      scf.yield(%4 : !i1)
    }
    %5 : !i1 = scf.if(%0 : !i1) {
      %6 : !i1 = arith.constant() ["value" = true]
      %7 : !i1 = arith.constant() ["value" = true]
      %8 : !i1 = arith.andi(%7 : !i1, %6 : !i1)
      scf.yield(%8 : !i1)
    }
    func.return(%1 : !i1, %5 : !i1)
  }
}
"""


def analyze_rewriting_locality(module: ModuleOp) -> list[int]:
    """Analyze locality of rewriting operations in a module.

    Args:
        module (Operation): The module to analyze.
    """
    module.attributes["rewriting_locality"] = IntegerAttr.from_int_and_width(
        0, 32)
    setattr(module, "dependent_ops", {module})
    for op in reversed(module.ops):
        analyze_op(op)

    rewriting_localities: list[int] = []

    def walk_all_ops(op: Operation):
        if "rewriting_locality" in op.attributes:
            rewriting_localities.append(
                op.attributes["rewriting_locality"].value.data)
        else:
            assert False, f"rewriting_locality not set for {op}"
        for region in op.regions:
            for block in region.blocks:
                for nested_op in block.ops:
                    walk_all_ops(nested_op)

    walk_all_ops(module)

    return rewriting_localities


def analyze_op(op: Operation) -> set[Operation]:
    """
    returns the number of other ops that have to be touched when this op is rewritten
    """
    dependent_ops = set[Operation]()

    if "rewriting_locality" in op.attributes:
        # return op.attributes["rewriting_locality"].value.data
        return getattr(op, "dependent_ops")
    for result in op.results:
        for use in result.uses:
            dependent_ops = dependent_ops | analyze_op(use.operation)

    # take rebuilding parent regions into account
    if parent := op.parent_op():
        dependent_ops = dependent_ops | analyze_op(parent)

    dependent_ops.add(op)
    op.attributes["rewriting_locality"] = IntegerAttr.from_int_and_width(
        len(dependent_ops) - 1, 32)
    setattr(op, "dependent_ops", dependent_ops)

    # Start analysis for all nested ops
    for region in op.regions:
        for block in region.blocks:
            for nested_op in reversed(block.ops):
                analyze_op(nested_op)

    # return count
    return dependent_ops


def main():
    arg_parser = argparse.ArgumentParser(
        prog='top', description='Show top lines from each file')
    arg_parser.add_argument("input_file",
                            type=argparse.FileType('r'),
                            nargs="?",
                            default=stdin)
    arg_parser.add_argument("--print", default=False, action='store_true')

    args = arg_parser.parse_args()

    ctx = MLContext()
    Builtin(ctx)
    Func(ctx)
    Arith(ctx)
    scf.Scf(ctx)
    onnx.Onnx(ctx)
    stencil.Stencil(ctx)

    parser = Parser(ctx, args.input_file.read(), source=Parser.Source.XDSL)
    module: Operation = parser.parse_op()

    assert module
    rewriting_localities = analyze_rewriting_locality(module)

    if args.print:
        printer = Printer()
        printer.print_op(module)

    # print(f"rewriting_localities: {rewriting_localities}")
    # print(f"max: {max(rewriting_localities)}")
    # print(f"mean: {statistics.mean(rewriting_localities)}")
    # print(f"median: {statistics.median(rewriting_localities)}")
    # print(f"stdev: {statistics.stdev(rewriting_localities)}")

    print(
        f"{max(rewriting_localities)};{statistics.mean(rewriting_localities)};{statistics.median(rewriting_localities)};{statistics.stdev(rewriting_localities)};{len(rewriting_localities)}"
    )
    # print(rewriting_localities)


if __name__ == "__main__":
    main()