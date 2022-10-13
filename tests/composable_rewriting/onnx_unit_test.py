from __future__ import annotations
from io import StringIO
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
import xdsl.dialects.onnx.dialect as onnx
import xdsl.dialects.IRUtils.dialect as irutils
import xdsl.dialects.pdl.dialect as pdl
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
import difflib
import os


def apply_dyn_strategy_and_compare(program: str, expected_program: str,
                                   strategy_name: str):
    ctx = MLContext()
    Builtin(ctx)
    Func(ctx)
    Arith(ctx)
    scf.Scf(ctx)
    irutils.IRUtils(ctx)
    pdl.PDL(ctx)
    match.Match(ctx)
    rewrite.Rewrite(ctx)
    elevate.Elevate(ctx)
    onnx.Onnx(ctx)

    # fetch strategies.xdsl file
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    strategy_file = open(
        os.path.join(__location__, '../../xdsl/dialects/onnx/strategies.xdsl'))
    strategy_string = strategy_file.read()

    ir_parser = Parser(ctx, program, source=Parser.Source.MLIR)
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


# From onnx-mlir
# To get the generic representation: onnx-mlir-opt test/mlir/onnx/onnx_canonicalization.mlir -split-input-file -mlir-print-op-generic


def test_matmul_add_fused():
    #// CHECK-NEXT: %{{[0-9]+}} = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>

    before = \
"""
"func.func"() ({
^bb0(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>, %arg2: tensor<10x10xf32>):
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%1) : (tensor<10x10xf32>) -> ()
}) {function_type = (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>, sym_name = "test_matmul_add_fused"} : () -> ()
"""
    fused = \
"""func.func() ["function_type" = !fun<[!tensor<[10 : !index, 10 : !index], !f32>, !tensor<[10 : !index, 10 : !index], !f32>, !tensor<[10 : !index, 10 : !index], !f32>], [!tensor<[10 : !index, 10 : !index], !f32>]>, "sym_name" = "test_matmul_add_fused"] {
^0(%0 : !tensor<[10 : !index, 10 : !index], !f32>, %1 : !tensor<[10 : !index, 10 : !index], !f32>, %2 : !tensor<[10 : !index, 10 : !index], !f32>):
  %3 : !tensor<[10 : !index, 10 : !index], !f32> = onnx.Gemm(%0 : !tensor<[10 : !index, 10 : !index], !f32>, %1 : !tensor<[10 : !index, 10 : !index], !f32>, %2 : !tensor<[10 : !index, 10 : !index], !f32>) ["alpha" = 1.0 : !f32, "beta" = 1.0 : !f32, "transA" = 0 : !i64, "transB" = 0 : !i64]
  func.return(%3 : !tensor<[10 : !index, 10 : !index], !f32>)
}
"""

    # similar rewrite, different traversals:
    apply_dyn_strategy_and_compare(program=before,
                                   expected_program=fused,
                                   strategy_name="onnx_opt_pass")


if __name__ == "__main__":
    test_matmul_add_fused()