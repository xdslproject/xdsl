// RUN: mlir-opt %s --linalg-generalize-named-ops  | xdsl-opt | filecheck %s

// RUN: xdsl-run %s | filecheck %s
// RUN: xdsl-opt %s -p convert-onnx-to-linalg
// RUN: xdsl-run %s --symbol main_graph --args "dense<1.0> : tensor<1x1x28x28xf32>" --verbose | filecheck %s
// RUN: xdsl-opt %s -p "convert-onnx-to-linalg,mlir-opt[linalg-named-op-conversion]" | xdsl-run | filecheck %s
// RUN: mlir-opt %s | xdsl-opt | filecheck %s

builtin.module attributes  {"llvm.data_layout" = "e-m:o-i64:64-i128:128-n32:64-S128", "llvm.target_triple" = "arm64-apple-darwin23.1.0", "onnx-mlir.symbol-postfix" = "mnist"} {
  func.func @main_graph(%arg0 : tensor<1x1x28x28xf32>) -> tensor<1x10xf32>  attributes {"input_names" = ["input.1"], "output_names" = ["19"]}{
    %0 = ml_program.global_load_const @onnx_constant_1 : tensor<2xi64>
    %1 = ml_program.global_load_const @onnx_constant_2 : tensor<10x1x5x5xf32>
    %2 = ml_program.global_load_const @onnx_constant_3 : tensor<10xf32>
    %3 = ml_program.global_load_const @onnx_constant_4 : tensor<20x10x5x5xf32>
    %4 = ml_program.global_load_const @onnx_constant_5 : tensor<20xf32>
    %5 = ml_program.global_load_const @onnx_constant_6 : tensor<50x320xf32>
    %6 = ml_program.global_load_const @onnx_constant_7 : tensor<50xf32>
    %7 = ml_program.global_load_const @onnx_constant_8 : tensor<10x50xf32>
    %8 = ml_program.global_load_const @onnx_constant_9 : tensor<10xf32>
    %9 = tensor.empty() : tensor<1x10x24x24xf32>
    %10 = linalg.conv_2d_nchw_fchw {"dilations" = dense<1> : tensor<2xi64>, "strides" = dense<1> : tensor<2xi64>} ins(%arg0, %1 : tensor<1x1x28x28xf32>, tensor<10x1x5x5xf32>) outs(%9 : tensor<1x10x24x24xf32>) -> tensor<1x10x24x24xf32>
    %11 = linalg.broadcast ins(%2:tensor<10xf32>) outs(%10:tensor<1x10x24x24xf32>) dimensions = [0, 2, 3]
    %12 = linalg.add ins(%11, %10 : tensor<1x10x24x24xf32>, tensor<1x10x24x24xf32>) outs(%10 : tensor<1x10x24x24xf32>) -> tensor<1x10x24x24xf32>
    %13 = tensor.empty() : tensor<1x10x24x24xf32>
    %14 = arith.constant 0.000000e+00 : f32
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12 : tensor<1x10x24x24xf32>) outs(%13 : tensor<1x10x24x24xf32>) {
    ^0(%16 : f32, %17 : f32):
      %18 = arith.maximumf %16, %14 : f32
      linalg.yield %18 : f32
    } -> tensor<1x10x24x24xf32>
    %19 = tensor.empty() : tensor<2x2xf32>
    %20 = tensor.empty() : tensor<1x10x12x12xf32>
    %21 = arith.constant -1.000000e+308 : f64
    %22 = linalg.fill ins(%21 : f64) outs(%20 : tensor<1x10x12x12xf32>) -> tensor<1x10x12x12xf32>
    %23 = linalg.pooling_nchw_max {"dilations" = dense<1> : tensor<2xi64>, "strides" = dense<2> : tensor<2xi64>} ins(%15, %19 : tensor<1x10x24x24xf32>, tensor<2x2xf32>) outs(%22 : tensor<1x10x12x12xf32>) -> tensor<1x10x12x12xf32>
    %24 = tensor.empty() : tensor<1x20x8x8xf32>
    %25 = linalg.conv_2d_nchw_fchw {"dilations" = dense<1> : tensor<2xi64>, "strides" = dense<1> : tensor<2xi64>} ins(%23, %3 : tensor<1x10x12x12xf32>, tensor<20x10x5x5xf32>) outs(%24 : tensor<1x20x8x8xf32>) -> tensor<1x20x8x8xf32>
    %26 = linalg.broadcast ins(%4:tensor<20xf32>) outs(%25:tensor<1x20x8x8xf32>) dimensions = [0, 2, 3]
    %27 = linalg.add ins(%26, %25 : tensor<1x20x8x8xf32>, tensor<1x20x8x8xf32>) outs(%25 : tensor<1x20x8x8xf32>) -> tensor<1x20x8x8xf32>
    %28 = tensor.empty() : tensor<1x20x8x8xf32>
    %29 = arith.constant 0.000000e+00 : f32
    %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27 : tensor<1x20x8x8xf32>) outs(%28 : tensor<1x20x8x8xf32>) {
    ^1(%31 : f32, %32 : f32):
      %33 = arith.maximumf %31, %29 : f32
      linalg.yield %33 : f32
    } -> tensor<1x20x8x8xf32>
    %34 = tensor.empty() : tensor<2x2xf32>
    %35 = tensor.empty() : tensor<1x20x4x4xf32>
    %36 = arith.constant -1.000000e+308 : f64
    %37 = linalg.fill ins(%36 : f64) outs(%35 : tensor<1x20x4x4xf32>) -> tensor<1x20x4x4xf32>
    %38 = linalg.pooling_nchw_max {"dilations" = dense<1> : tensor<2xi64>, "strides" = dense<2> : tensor<2xi64>} ins(%30, %34 : tensor<1x20x8x8xf32>, tensor<2x2xf32>) outs(%37 : tensor<1x20x4x4xf32>) -> tensor<1x20x4x4xf32>
    %39 = tensor.reshape %38(%0) : (tensor<1x20x4x4xf32>, tensor<2xi64>) -> tensor<1x320xf32>
    %40 = tensor.empty() : tensor<320x50xf32>
    %41 = linalg.transpose ins(%5:tensor<50x320xf32>) outs(%40:tensor<320x50xf32>) permutation = [1, 0]
    %42 = tensor.empty() : tensor<1x50xf32>
    %43 = linalg.matmul ins(%39, %41 : tensor<1x320xf32>, tensor<320x50xf32>) outs(%42 : tensor<1x50xf32>) -> tensor<1x50xf32>
    %44 = linalg.broadcast ins(%6:tensor<50xf32>) outs(%43:tensor<1x50xf32>) dimensions = [0]
    %45 = linalg.add ins(%44, %43 : tensor<1x50xf32>, tensor<1x50xf32>) outs(%43 : tensor<1x50xf32>) -> tensor<1x50xf32>
    %46 = tensor.empty() : tensor<1x50xf32>
    %47 = arith.constant 0.000000e+00 : f32
    %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%45 : tensor<1x50xf32>) outs(%46 : tensor<1x50xf32>) {
    ^2(%49 : f32, %50 : f32):
      %51 = arith.maximumf %49, %47 : f32
      linalg.yield %51 : f32
    } -> tensor<1x50xf32>
    %52 = tensor.empty() : tensor<50x10xf32>
    %53 = linalg.transpose ins(%7:tensor<10x50xf32>) outs(%52:tensor<50x10xf32>) permutation = [1, 0]
    %54 = tensor.empty() : tensor<1x10xf32>
    %55 = linalg.matmul ins(%48, %53 : tensor<1x50xf32>, tensor<50x10xf32>) outs(%54 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %56 = linalg.broadcast ins(%8:tensor<10xf32>) outs(%55:tensor<1x10xf32>) dimensions = [0]
    %57 = linalg.add ins(%56, %55 : tensor<1x10xf32>, tensor<1x10xf32>) outs(%55 : tensor<1x10xf32>) -> tensor<1x10xf32>
    func.return %57 : tensor<1x10xf32>
  }
  "onnx.EntryPoint"() {"func" = @main_graph} : () -> ()
  ml_program.global private @onnx_constant_1(dense<[-1, 320]> : tensor<2xi64>) : tensor<2xi64>
  ml_program.global private @onnx_constant_2(dense<8.536267e-02> : tensor<10x1x5x5xf32>) : tensor<10x1x5x5xf32>
  ml_program.global private @onnx_constant_3(dense<[1.437707e-01, 4.253821e-02, -4.443112e-02, 6.546726e-02, -2.458754e-02, -5.265858e-03, 1.039173e-01, -3.793048e-01, 4.960973e-02, 1.660075e-01]> : tensor<10xf32>) : tensor<10xf32>
  ml_program.global private @onnx_constant_4(dense<1.198930e-01> : tensor<20x10x5x5xf32>) : tensor<20x10x5x5xf32>
  ml_program.global private @onnx_constant_5(dense<[6.413303e-02, -5.045709e-02, 2.340628e-02, 2.116011e-02, 6.684135e-02, 2.236916e-02, 5.442252e-02, 1.337397e-03, -3.487194e-02, 3.570483e-02, -5.345386e-02, -5.606352e-03, -5.570797e-02, -3.902349e-02, -5.761088e-02, 3.830317e-02, -6.468914e-02, 6.983487e-02, -1.061398e-02, -2.971869e-02]> : tensor<20xf32>) : tensor<20xf32>
  ml_program.global private @onnx_constant_6(dense<4.372590e-02> : tensor<50x320xf32>) : tensor<50x320xf32>
  ml_program.global private @onnx_constant_7(dense<[-4.073842e-02, 8.931790e-03, -1.663272e-02, 4.353305e-02, -2.659053e-02, 1.494592e-02, 2.349606e-02, -5.280812e-02, 2.866207e-02, 5.509599e-02, -4.067043e-02, 4.658677e-02, 4.150655e-02, 2.584196e-02, 1.757336e-02, -5.987659e-03, 3.132654e-02, -2.453315e-02, 6.139632e-02, -5.594915e-02, 7.188448e-02, 3.004006e-02, 5.483749e-03, 3.162697e-02, 5.479893e-02, -1.113903e-02, -4.275452e-02, -2.675450e-02, -3.154664e-02, 5.657679e-02, -3.933117e-02, -2.314896e-02, -1.980006e-03, -2.770574e-02, -4.384881e-02, 4.202367e-02, 5.017417e-02, -1.182935e-02, -1.193583e-02, 2.243767e-02, 4.175425e-02, -2.241457e-02, 3.025757e-02, -4.480691e-02, -4.015515e-02, -1.344138e-02, -2.460308e-02, 3.854027e-02, -2.318060e-03, 1.823421e-02]> : tensor<50xf32>) : tensor<50xf32>
  ml_program.global private @onnx_constant_8(dense<5.608376e-01> : tensor<10x50xf32>) : tensor<10x50xf32>
  ml_program.global private @onnx_constant_9(dense<[-1.636385e-01, -2.030399e-02, 1.214530e-01, 1.311745e-01, -1.271445e-01, 2.372866e-02, -1.297138e-01, -1.380079e-01, 1.668115e-01, -1.186569e-01]> : tensor<10xf32>) : tensor<10xf32>
}

