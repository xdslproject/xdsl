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
        os.path.join(__location__,
                     '../../../xdsl/dialects/onnx/strategies.xdsl'))
    strategy_string = strategy_file.read()

    ir_parser = Parser(ctx, program, source=Parser.Source.MLIR)
    ir_module: Operation = ir_parser.parse_op()
    imm_ir_module: IOp = get_immutable_copy(ir_module)

    strat_parser = Parser(ctx, strategy_string)
    strat_module: Operation = strat_parser.parse_op()
    elevate_interpreter = interpreter.ElevateInterpreter()

    elevate_interpreter.register_native_strategy(GarbageCollect,
                                                 "garbage_collect")
    elevate_interpreter.register_native_strategy(id, "id")
    elevate_interpreter.register_native_strategy(debug, "debug")

    strategies = elevate_interpreter.get_strategy(strat_module)
    strategy = strategies[strategy_name]

    rr = strategy.apply(imm_ir_module)
    assert rr.isSuccess()

    # for debugging
    printer = Printer(target=Printer.Target.MLIR)
    print(f'Result after applying "{strategy}":')
    printer.print_op(rr.result_op.get_mutable_copy())

    file = StringIO("")
    printer = Printer(stream=file, target=Printer.Target.MLIR)
    printer.print_op(rr.result_op.get_mutable_copy())

    diff = difflib.Differ().compare(file.getvalue().splitlines(True),
                                    expected_program.splitlines(True))
    if file.getvalue().strip() != expected_program.strip():
        print("Did not get expected output! Diff:")
        print(''.join(diff))
        assert False


def parse(program: str):
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

    parser = Parser(ctx, program, source=Parser.Source.MLIR)
    module: Operation = parser.parse_op()

    printer = Printer(target=Printer.Target.MLIR)
    printer.print_op(module)


# From onnx-mlir
# To get the generic representation: onnx-mlir-opt test/mlir/onnx/onnx_canonicalization.mlir -split-input-file -mlir-print-op-generic


def test_fuse_bert_attention_layer():
    unfused_attention_layer = \
"""
"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : tensor<1x3x16xf32>, %1 : tensor<1x3x16xf32>, %2 : tensor<1x3xi64>):
    %3 = "onnx.Constant"() {"value" = dense<[1.0]> : tensor<16xf32>} : () -> tensor<16xf32>
    %4 = "onnx.Constant"() {"value" = dense<[1.0]> : tensor<16xf32>} : () -> tensor<16xf32>
    %5 = "onnx.Constant"() {"value" = dense<[1.0]> : tensor<16x16xf32>} : () -> tensor<16x16xf32>
    %6 = "onnx.Constant"() {"value" = dense<[1.0]> : tensor<16x16xf32>} : () -> tensor<16x16xf32>
    %7 = "onnx.Constant"() {"value" = dense<[1.0]> : tensor<16x16xf32>} : () -> tensor<16x16xf32>
    %8 = "onnx.Constant"() {"value" = dense<[1.0]> : tensor<16x16xf32>} : () -> tensor<16x16xf32>
    %9 = "onnx.Constant"() {"value" = dense<[1.0]> : tensor<16xf32>} : () -> tensor<16xf32>
    %10 = "onnx.Constant"() {"value" = dense<[1.0]> : tensor<16xf32>} : () -> tensor<16xf32>
    %11 = "onnx.Constant"() {"value" = dense<[1.0]> : tensor<16xf32>} : () -> tensor<16xf32>
    %12 = "onnx.Constant"() {"value" = dense<[1.0]> : tensor<16xf32>} : () -> tensor<16xf32>
    %13 = "onnx.Constant"() {"value" = dense<[2.82842708]> : tensor<1xf32>} : () -> tensor<1xf32>
    %14 = "onnx.Constant"() {"value" = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
    %15 = "onnx.Constant"() {"value" = dense<[-10000.0]> : tensor<1xf32>} : () -> tensor<1xf32>
    %16 = "onnx.Constant"() {"value" = dense<[0, 0, 2, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %17 = "onnx.Constant"() {"value" = dense<[0, 0, 2, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
    %18 = "onnx.Constant"() {"value" = dense<[0, 0, 16]> : tensor<3xi64>} : () -> tensor<3xi64>
    %19 = "onnx.Constant"() {"value" = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %20 = "onnx.Constant"() {"value" = dense<[2]> : tensor<1xi64>} : () -> tensor<1xi64>
    %21 = "onnx.Add"(%0, %1) {"onnx_node_name" = "add_layernorm"} : (tensor<1x3x16xf32>, tensor<1x3x16xf32>) -> tensor<1x3x16xf32>
    %22 = "onnx.Custom"(%21, %3, %4) {"axis" = -1 : si64, "domain_name" = "", "epsion" = 9.99999974e-06 : f32, "function_name" = "LayerNormalization", "onnx_node_name" = "layernorm"} : (tensor<1x3x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<*xf32>
    %23 = "onnx.MatMul"(%22, %5) {"onnx_node_name" = "matmul_q"} : (tensor<*xf32>, tensor<16x16xf32>) -> tensor<*xf32>
    %24 = "onnx.Add"(%23, %9) {"onnx_node_name" = "add_q"} : (tensor<*xf32>, tensor<16xf32>) -> tensor<*xf32>
    %25 = "onnx.Reshape"(%24, %16) {"onnx_node_name" = "reshape_q"} : (tensor<*xf32>, tensor<4xi64>) -> tensor<*xf32>
    %26 = "onnx.Transpose"(%25) {"onnx_node_name" = "transpose_q", "perm" = [0 : i64, 2 : i64, 1 : i64, 3 : i64]} : (tensor<*xf32>) -> tensor<*xf32>
    %27 = "onnx.MatMul"(%22, %6) {"onnx_node_name" = "matmul_k"} : (tensor<*xf32>, tensor<16x16xf32>) -> tensor<*xf32>
    %28 = "onnx.Add"(%27, %10) {"onnx_node_name" = "add_k"} : (tensor<*xf32>, tensor<16xf32>) -> tensor<*xf32>
    %29 = "onnx.Reshape"(%28, %16) {"onnx_node_name" = "reshape_k"} : (tensor<*xf32>, tensor<4xi64>) -> tensor<*xf32>
    %30 = "onnx.Transpose"(%29) {"onnx_node_name" = "transpose_k", "perm" = [0 : i64, 2 : i64, 3 : i64, 1 : i64]} : (tensor<*xf32>) -> tensor<*xf32>
    %31 = "onnx.Unsqueeze"(%2, %19) {"onnx_node_name" = "unsqueeze0"} : (tensor<1x3xi64>, tensor<1xi64>) -> tensor<1x1x3xi64>
    %32 = "onnx.Unsqueeze"(%31, %20) {"onnx_node_name" = "unsqueeze1"} : (tensor<1x1x3xi64>, tensor<1xi64>) -> tensor<1x1x1x3xi64>
    %33 = "onnx.Cast"(%32) {"onnx_node_name" = "cast", "to" = f32} : (tensor<1x1x1x3xi64>) -> tensor<1x1x1x3xf32>
    %34 = "onnx.Sub"(%14, %33) {"onnx_node_name" = "sub"} : (tensor<1xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x1x3xf32>
    %35 = "onnx.Mul"(%34, %15) {"onnx_node_name" = "mul_mask"} : (tensor<1x1x1x3xf32>, tensor<1xf32>) -> tensor<1x1x1x3xf32>
    %36 = "onnx.MatMul"(%26, %30) {"onnx_node_name" = "matmul_qk"} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %37 = "onnx.Div"(%36, %13) {"onnx_node_name" = "div_qk"} : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
    %38 = "onnx.Add"(%37, %35) {"onnx_node_name" = "add_qk"} : (tensor<*xf32>, tensor<1x1x1x3xf32>) -> tensor<*xf32>
    %39 = "onnx.Softmax"(%38) {"axis" = 3 : si64, "onnx_node_name" = "softmax_qk", "onnx_opset" = 0 : si64} : (tensor<*xf32>) -> tensor<*xf32>
    %40 = "onnx.MatMul"(%22, %7) {"onnx_node_name" = "matmul_v"} : (tensor<*xf32>, tensor<16x16xf32>) -> tensor<*xf32>
    %41 = "onnx.Add"(%40, %11) {"onnx_node_name" = "add_v"} : (tensor<*xf32>, tensor<16xf32>) -> tensor<*xf32>
    %42 = "onnx.Reshape"(%41, %17) {"onnx_node_name" = "reshape_v"} : (tensor<*xf32>, tensor<4xi64>) -> tensor<*xf32>
    %43 = "onnx.Transpose"(%42) {"onnx_node_name" = "transpose_v", "perm" = [0 : i64, 2 : i64, 1 : i64, 3 : i64]} : (tensor<*xf32>) -> tensor<*xf32>
    %44 = "onnx.MatMul"(%39, %43) {"onnx_node_name" = "matmul_qkv_1"} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %45 = "onnx.Transpose"(%44) {"onnx_node_name" = "transpose_qkv", "perm" = [0 : i64, 2 : i64, 1 : i64, 3 : i64]} : (tensor<*xf32>) -> tensor<*xf32>
    %46 = "onnx.Reshape"(%45, %18) {"onnx_node_name" = "reshape_qkv"} : (tensor<*xf32>, tensor<3xi64>) -> tensor<*xf32>
    %47 = "onnx.MatMul"(%46, %8) {"onnx_node_name" = "matmul_qkv_2"} : (tensor<*xf32>, tensor<16x16xf32>) -> tensor<*xf32>
    %48 = "onnx.Add"(%47, %22) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %49 = "onnx.Add"(%48, %12) : (tensor<*xf32>, tensor<16xf32>) -> tensor<*xf32>
    %50 = "onnx.Custom"(%49, %3, %4) {"axis" = -1 : si64, "domain_name" = "", "epsion" = 9.99999974e-06 : f32, "function_name" = "LayerNormalization", "onnx_node_name" = "layernorm2"} : (tensor<*xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1x3x16xf32>
    "func.return"(%50) : (tensor<1x3x16xf32>) -> ()
  }) {"function_type" = (tensor<1x3x16xf32>, tensor<1x3x16xf32>, tensor<1x3xi64>) -> tensor<1x3x16xf32>, "input_names" = ["input_1", "input_2", "input_mask"], "output_names" = ["output"], "sym_name" = "main_graph"} : () -> ()
  "onnx.EntryPoint"() {"func" = @main_graph} : () -> ()
}) : () -> ()
"""


    fused_attention_layer = \
"""
"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : tensor<1x3x16xf32>, %1 : tensor<1x3x16xf32>, %2 : tensor<1x3xi64>):
    %3 = "onnx.Constant"() {"value" = dense<[1.0]> : tensor<16xf32>} : () -> tensor<16xf32>
    %4 = "onnx.Constant"() {"value" = dense<[1.0]> : tensor<16xf32>} : () -> tensor<16xf32>
    %5 = "onnx.Constant"() {"value" = dense<[1.0]> : tensor<16xf32>} : () -> tensor<16xf32>
    %6 = "onnx.Add"(%0, %1) {"onnx_node_name" = "add_layernorm"} : (tensor<1x3x16xf32>, tensor<1x3x16xf32>) -> tensor<1x3x16xf32>
    %7 = "onnx.Custom"(%6, %3, %4) {"axis" = -1 : si64, "domain_name" = "", "epsion" = 9.99999974e-06 : f32, "function_name" = "LayerNormalization", "onnx_node_name" = "layernorm"} : (tensor<1x3x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<*xf32>
    %8 = "onnx.Cast"(%2) {"onnx_node_name" = "Cast3", "to" = i32} : (tensor<1x3xi64>) -> tensor<1x3xi32>
    %9 = "onnx.Constant"() {"value" = dense<[1.0]> : tensor<16x48xf32>} : () -> tensor<16x48xf32>
    %10 = "onnx.Constant"() {"value" = dense<[1.0]> : tensor<48xf32>} : () -> tensor<48xf32>
    %11 = "onnx.Custom"(%7, %9, %10, %8) {"domain_name" = "com.microsoft", "function_name" = "Attention", "num_heads" = 2 : si64, "onnx_node_name" = "Attention_0"} : (tensor<*xf32>, tensor<16x48xf32>, tensor<48xf32>, tensor<1x3xi32>) -> tensor<*xf32>
    %12 = "onnx.MatMul"(%11, %11) {"onnx_node_name" = "matmul_qkv_2"} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %13 = "onnx.Custom"(%7, %12, %3, %4, %5) {"domain_name" = "com.microsoft", "epsilon" = 9.99999996e-13 : f32, "function_name" = "SkipLayerNormalization", "onnx_node_name" = "SkipLayerNorm_AddBias_0"} : (tensor<*xf32>, tensor<*xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1x3x16xf32>
    "func.return"(%13) : (tensor<1x3x16xf32>) -> ()
  }) {"function_type" = (tensor<1x3x16xf32>, tensor<1x3x16xf32>, tensor<1x3xi64>) -> tensor<1x3x16xf32>, "input_names" = ["input_1", "input_2", "input_mask"], "output_names" = ["output"], "sym_name" = "main_graph"} : () -> ()
  "onnx.EntryPoint"() {"func" = @main_graph} : () -> ()
}) : () -> ()
"""

    apply_dyn_strategy_and_compare(
        program=unfused_attention_layer,
        expected_program=fused_attention_layer,
        strategy_name="top_to_bottom_fuse_attention_layer")


if __name__ == "__main__":
    test_fuse_bert_attention_layer()