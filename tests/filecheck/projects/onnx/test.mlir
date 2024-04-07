// RUN: mlir-opt %s --linalg-generalize-named-ops  | xdsl-opt | filecheck %s

module attributes  {"llvm.data_layout" = "e-m:o-i64:64-i128:128-n32:64-S128", "llvm.target_triple" = "arm64-apple-darwin23.1.0", "onnx-mlir.symbol-postfix" = "mnist"} {
  func.func @main_graph(%arg0 : tensor<1x1x28x28xf32>) -> tensor<1x10xf32>  attributes {"input_names" = ["input.1"], "output_names" = ["19"]}{
    %0 = ml_program.global_load_const @onnx_constant_1 : tensor<2xi64>
    %1 = ml_program.global_load_const @onnx_constant_2 : tensor<10x1x5x5xf32>
    %2 = ml_program.global_load_const @onnx_constant_3 : tensor<1x10x24x24xf32>
    %3 = ml_program.global_load_const @onnx_constant_4 : tensor<20x10x5x5xf32>
    %4 = ml_program.global_load_const @onnx_constant_5 : tensor<20xf32>
    %5 = ml_program.global_load_const @onnx_constant_6 : tensor<50x320xf32>
    %6 = ml_program.global_load_const @onnx_constant_7 : tensor<50xf32>
    %7 = ml_program.global_load_const @onnx_constant_8 : tensor<10x50xf32>
    %8 = ml_program.global_load_const @onnx_constant_9 : tensor<10xf32>
    %9 = tensor.empty() : tensor<1x10x24x24xf32>
    %10 = linalg.conv_2d_nchw_fchw {"dilations" = dense<1> : tensor<2xi64>, "strides" = dense<1> : tensor<2xi64>} ins(%arg0, %1 : tensor<1x1x28x28xf32>, tensor<10x1x5x5xf32>) outs(%9 : tensor<1x10x24x24xf32>) -> tensor<1x10x24x24xf32>
    %11 = linalg.add ins(%2, %10 : tensor<1x10x24x24xf32>, tensor<1x10x24x24xf32>) outs(%10 : tensor<1x10x24x24xf32>) -> tensor<1x10x24x24xf32>
    %12 = tensor.empty() : tensor<1x10x24x24xf32>
    %13 = arith.constant 0.000000e+00 : f32
    %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11 : tensor<1x10x24x24xf32>) outs(%12 : tensor<1x10x24x24xf32>) {
    ^0(%15 : f32, %16 : f32):
      %17 = arith.maximumf %15, %13 : f32
      linalg.yield %17 : f32
    } -> tensor<1x10x24x24xf32>
    %18 = tensor.empty() : tensor<2x2xf32>
    %19 = tensor.empty() : tensor<1x10x12x12xf32>
    %20 = arith.constant -1.000000e+308 : f64
    %21 = linalg.fill ins(%20 : f64) outs(%19 : tensor<1x10x12x12xf32>) -> tensor<1x10x12x12xf32>
    %22 = linalg.pooling_nchw_max {"dilations" = dense<1> : tensor<2xi64>, "strides" = dense<2> : tensor<2xi64>} ins(%14, %18 : tensor<1x10x24x24xf32>, tensor<2x2xf32>) outs(%21 : tensor<1x10x12x12xf32>) -> tensor<1x10x12x12xf32>
    %23 = tensor.empty() : tensor<1x20x8x8xf32>
    %24 = linalg.conv_2d_nchw_fchw {"dilations" = dense<1> : tensor<2xi64>, "strides" = dense<1> : tensor<2xi64>} ins(%22, %3 : tensor<1x10x12x12xf32>, tensor<20x10x5x5xf32>) outs(%23 : tensor<1x20x8x8xf32>) -> tensor<1x20x8x8xf32>
    %25 = linalg.add ins(%4, %24 : tensor<20xf32>, tensor<1x20x8x8xf32>) outs(%24 : tensor<1x20x8x8xf32>) -> tensor<1x20x8x8xf32>
    %26 = tensor.empty() : tensor<1x20x8x8xf32>
    %27 = arith.constant 0.000000e+00 : f32
    %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel"]} ins(%25 : tensor<1x20x8x8xf32>) outs(%26 : tensor<1x20x8x8xf32>) {
    ^1(%29 : f32, %30 : f32):
      %31 = arith.maximumf %29, %27 : f32
      linalg.yield %31 : f32
    } -> tensor<1x20x8x8xf32>
    %32 = tensor.empty() : tensor<2x2xf32>
    %33 = tensor.empty() : tensor<1x20x4x4xf32>
    %34 = arith.constant -1.000000e+308 : f64
    %35 = linalg.fill ins(%34 : f64) outs(%33 : tensor<1x20x4x4xf32>) -> tensor<1x20x4x4xf32>
    %36 = linalg.pooling_nchw_max {"dilations" = dense<1> : tensor<2xi64>, "strides" = dense<2> : tensor<2xi64>} ins(%28, %32 : tensor<1x20x8x8xf32>, tensor<2x2xf32>) outs(%35 : tensor<1x20x4x4xf32>) -> tensor<1x20x4x4xf32>
    %37 = tensor.reshape %36(%0) : (tensor<1x20x4x4xf32>, tensor<2xi64>) -> tensor<1x320xf32>
    %38 = tensor.empty() : tensor<320x50xf32>
    %39 = linalg.transpose ins(%5:tensor<50x320xf32>) outs(%38:tensor<320x50xf32>) permutation = [1, 0]
    %40 = tensor.empty() : tensor<1x50xf32>
    %41 = linalg.matmul ins(%37, %39 : tensor<1x320xf32>, tensor<320x50xf32>) outs(%40 : tensor<1x50xf32>) -> tensor<1x50xf32>
    %42 = linalg.add ins(%41, %6 : tensor<1x50xf32>, tensor<50xf32>) outs(%41 : tensor<1x50xf32>) -> tensor<1x50xf32>
    %43 = tensor.empty() : tensor<1x50xf32>
    %44 = arith.constant 0.000000e+00 : f32
    %45 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%42 : tensor<1x50xf32>) outs(%43 : tensor<1x50xf32>) {
    ^2(%46 : f32, %47 : f32):
      %48 = arith.maximumf %46, %44 : f32
      linalg.yield %48 : f32
    } -> tensor<1x50xf32>
    %49 = tensor.empty() : tensor<50x10xf32>
    %50 = linalg.transpose ins(%7:tensor<10x50xf32>) outs(%49:tensor<50x10xf32>) permutation = [1, 0]
    %51 = tensor.empty() : tensor<1x10xf32>
    %52 = linalg.matmul ins(%45, %50 : tensor<1x50xf32>, tensor<50x10xf32>) outs(%51 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %53 = linalg.add ins(%52, %8 : tensor<1x10xf32>, tensor<10xf32>) outs(%52 : tensor<1x10xf32>) -> tensor<1x10xf32>
    func.return %53 : tensor<1x10xf32>
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



