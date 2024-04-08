// RUN: mlir-opt %s --linalg-generalize-named-ops  | xdsl-opt | filecheck %s

// RUN: xdsl-run %s | filecheck %s
// RUN: xdsl-opt %s -p convert-onnx-to-linalg
// RUN: xdsl-run %s --symbol main_graph --args "dense<1.0> : tensor<1x1x28x28xf32>" --verbose | filecheck %s
// RUN: xdsl-opt %s -p "convert-onnx-to-linalg,mlir-opt[linalg-named-op-conversion]" | xdsl-run | filecheck %s
// RUN: mlir-opt %s | xdsl-opt | filecheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> ()>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d3 * 2 + d5)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#map9 = affine_map<(d0, d1) -> (d0, d1)>
#map10 = affine_map<(d0, d1) -> (d1, d0)>
#map11 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map12 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map13 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map14 = affine_map<(d0, d1) -> (d1)>
module attributes {llvm.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128", llvm.target_triple = "arm64-apple-darwin23.1.0", "onnx-mlir.symbol-postfix" = "mnist"} {
  func.func @main_graph(%arg0: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> attributes {input_names = ["input.1"], output_names = ["19"]} {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %onnx_constant_1 = ml_program.global_load_const @onnx_constant_1 : tensor<2xi64>
    %onnx_constant_2 = ml_program.global_load_const @onnx_constant_2 : tensor<10x1x5x5xf32>
    %onnx_constant_3 = ml_program.global_load_const @onnx_constant_3 : tensor<10xf32>
    %onnx_constant_4 = ml_program.global_load_const @onnx_constant_4 : tensor<20x10x5x5xf32>
    %onnx_constant_5 = ml_program.global_load_const @onnx_constant_5 : tensor<20xf32>
    %onnx_constant_6 = ml_program.global_load_const @onnx_constant_6 : tensor<50x320xf32>
    %onnx_constant_7 = ml_program.global_load_const @onnx_constant_7 : tensor<50xf32>
    %onnx_constant_8 = ml_program.global_load_const @onnx_constant_8 : tensor<10x50xf32>
    %onnx_constant_9 = ml_program.global_load_const @onnx_constant_9 : tensor<10xf32>
    %0 = tensor.empty() : tensor<1x10x24x24xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %onnx_constant_2 : tensor<1x1x28x28xf32>, tensor<10x1x5x5xf32>) outs(%0 : tensor<1x10x24x24xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %34 = arith.mulf %in, %in_1 : f32
      %35 = arith.addf %out, %34 : f32
      linalg.yield %35 : f32
    } -> tensor<1x10x24x24xf32>
    %2 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%onnx_constant_3 : tensor<10xf32>) outs(%1 : tensor<1x10x24x24xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x10x24x24xf32>
    %3 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %1 : tensor<1x10x24x24xf32>, tensor<1x10x24x24xf32>) outs(%1 : tensor<1x10x24x24xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %34 = arith.addf %in, %in_1 : f32
      linalg.yield %34 : f32
    } -> tensor<1x10x24x24xf32>
    %4 = tensor.empty() : tensor<1x10x24x24xf32>
    %5 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<1x10x24x24xf32>) outs(%4 : tensor<1x10x24x24xf32>) {
    ^bb0(%in: f32, %out: f32):
      %34 = arith.maximumf %in, %cst_0 : f32
      linalg.yield %34 : f32
    } -> tensor<1x10x24x24xf32>
    %6 = tensor.empty() : tensor<2x2xf32>
    %7 = tensor.empty() : tensor<1x10x12x12xf32>
    %8 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst : f32) outs(%7 : tensor<1x10x12x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x10x12x12xf32>
    %9 = linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%5, %6 : tensor<1x10x24x24xf32>, tensor<2x2xf32>) outs(%8 : tensor<1x10x12x12xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %34 = arith.maximumf %out, %in : f32
      linalg.yield %34 : f32
    } -> tensor<1x10x12x12xf32>
    %10 = tensor.empty() : tensor<1x20x8x8xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%9, %onnx_constant_4 : tensor<1x10x12x12xf32>, tensor<20x10x5x5xf32>) outs(%10 : tensor<1x20x8x8xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %34 = arith.mulf %in, %in_1 : f32
      %35 = arith.addf %out, %34 : f32
      linalg.yield %35 : f32
    } -> tensor<1x20x8x8xf32>
    %12 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%onnx_constant_5 : tensor<20xf32>) outs(%11 : tensor<1x20x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x20x8x8xf32>
    %13 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12, %11 : tensor<1x20x8x8xf32>, tensor<1x20x8x8xf32>) outs(%11 : tensor<1x20x8x8xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %34 = arith.addf %in, %in_1 : f32
      linalg.yield %34 : f32
    } -> tensor<1x20x8x8xf32>
    %14 = tensor.empty() : tensor<1x20x8x8xf32>
    %15 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13 : tensor<1x20x8x8xf32>) outs(%14 : tensor<1x20x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %34 = arith.maximumf %in, %cst_0 : f32
      linalg.yield %34 : f32
    } -> tensor<1x20x8x8xf32>
    %16 = tensor.empty() : tensor<2x2xf32>
    %17 = tensor.empty() : tensor<1x20x4x4xf32>
    %18 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst : f32) outs(%17 : tensor<1x20x4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x20x4x4xf32>
    %19 = linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%15, %16 : tensor<1x20x8x8xf32>, tensor<2x2xf32>) outs(%18 : tensor<1x20x4x4xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %34 = arith.maximumf %out, %in : f32
      linalg.yield %34 : f32
    } -> tensor<1x20x4x4xf32>
    %reshape = tensor.reshape %19(%onnx_constant_1) : (tensor<1x20x4x4xf32>, tensor<2xi64>) -> tensor<1x320xf32>
    %20 = tensor.empty() : tensor<320x50xf32>
    %21 = linalg.generic {indexing_maps = [#map9, #map10], iterator_types = ["parallel", "parallel"]} ins(%onnx_constant_6 : tensor<50x320xf32>) outs(%20 : tensor<320x50xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<320x50xf32>
    %22 = tensor.empty() : tensor<1x50xf32>
    %23 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "reduction"]} ins(%reshape, %21 : tensor<1x320xf32>, tensor<320x50xf32>) outs(%22 : tensor<1x50xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %34 = arith.mulf %in, %in_1 : f32
      %35 = arith.addf %out, %34 : f32
      linalg.yield %35 : f32
    } -> tensor<1x50xf32>
    %24 = linalg.generic {indexing_maps = [#map14, #map9], iterator_types = ["parallel", "parallel"]} ins(%onnx_constant_7 : tensor<50xf32>) outs(%23 : tensor<1x50xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x50xf32>
    %25 = linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%24, %23 : tensor<1x50xf32>, tensor<1x50xf32>) outs(%23 : tensor<1x50xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %34 = arith.addf %in, %in_1 : f32
      linalg.yield %34 : f32
    } -> tensor<1x50xf32>
    %26 = tensor.empty() : tensor<1x50xf32>
    %27 = linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%25 : tensor<1x50xf32>) outs(%26 : tensor<1x50xf32>) {
    ^bb0(%in: f32, %out: f32):
      %34 = arith.maximumf %in, %cst_0 : f32
      linalg.yield %34 : f32
    } -> tensor<1x50xf32>
    %28 = tensor.empty() : tensor<50x10xf32>
    %29 = linalg.generic {indexing_maps = [#map9, #map10], iterator_types = ["parallel", "parallel"]} ins(%onnx_constant_8 : tensor<10x50xf32>) outs(%28 : tensor<50x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<50x10xf32>
    %30 = tensor.empty() : tensor<1x10xf32>
    %31 = linalg.generic {indexing_maps = [#map11, #map12, #map13], iterator_types = ["parallel", "parallel", "reduction"]} ins(%27, %29 : tensor<1x50xf32>, tensor<50x10xf32>) outs(%30 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %34 = arith.mulf %in, %in_1 : f32
      %35 = arith.addf %out, %34 : f32
      linalg.yield %35 : f32
    } -> tensor<1x10xf32>
    %32 = linalg.generic {indexing_maps = [#map14, #map9], iterator_types = ["parallel", "parallel"]} ins(%onnx_constant_9 : tensor<10xf32>) outs(%31 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x10xf32>
    %33 = linalg.generic {indexing_maps = [#map9, #map9, #map9], iterator_types = ["parallel", "parallel"]} ins(%32, %31 : tensor<1x10xf32>, tensor<1x10xf32>) outs(%31 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %34 = arith.addf %in, %in_1 : f32
      linalg.yield %34 : f32
    } -> tensor<1x10xf32>
    return %33 : tensor<1x10xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
  ml_program.global private @onnx_constant_1(dense<[-1, 320]> : tensor<2xi64>) : tensor<2xi64>
  ml_program.global private @onnx_constant_2(dense<0.0853626728> : tensor<10x1x5x5xf32>) : tensor<10x1x5x5xf32>
  ml_program.global private @onnx_constant_3(dense<[0.143770695, 0.0425382107, -0.0444311202, 0.0654672608, -0.02458754, -0.00526585802, 0.103917301, -0.379304796, 0.0496097282, 0.166007504]> : tensor<10xf32>) : tensor<10xf32>
  ml_program.global private @onnx_constant_4(dense<1.198930e-01> : tensor<20x10x5x5xf32>) : tensor<20x10x5x5xf32>
  ml_program.global private @onnx_constant_5(dense<[0.0641330332, -0.0504570901, 0.0234062802, 0.0211601108, 0.066841349, 0.0223691594, 0.0544225201, 0.00133739703, -0.0348719396, 0.0357048288, -0.0534538589, -0.00560635189, -0.0557079688, -0.0390234888, -0.0576108806, 0.0383031704, -0.064689137, 0.0698348731, -0.0106139798, -0.0297186896]> : tensor<20xf32>) : tensor<20xf32>
  ml_program.global private @onnx_constant_6(dense<4.372590e-02> : tensor<50x320xf32>) : tensor<50x320xf32>
  ml_program.global private @onnx_constant_7(dense<[-0.0407384187, 0.00893178954, -0.0166327208, 0.0435330495, -0.0265905298, 0.0149459196, 0.0234960597, -0.0528081208, 0.0286620706, 0.0550959893, -0.0406704284, 0.0465867706, 0.0415065512, 0.0258419607, 0.0175733604, -0.0059876591, 0.0313265398, -0.0245331507, 0.0613963194, -0.0559491515, 0.071884483, 0.0300400592, 0.00548374886, 0.0316269696, 0.0547989309, -0.0111390296, -0.0427545197, -2.675450e-02, -0.0315466411, 0.0565767884, -0.0393311717, -0.0231489595, -0.001980006, -0.0277057402, -0.0438488089, 0.0420236699, 0.0501741692, -0.0118293501, -0.0119358301, 0.0224376693, 0.0417542495, -0.0224145707, 0.0302575696, -0.0448069088, -0.04015515, -0.0134413801, -0.02460308, 0.0385402702, -2.318060e-03, 0.0182342101]> : tensor<50xf32>) : tensor<50xf32>
  ml_program.global private @onnx_constant_8(dense<0.560837626> : tensor<10x50xf32>) : tensor<10x50xf32>
  ml_program.global private @onnx_constant_9(dense<[-0.163638502, -0.0203039907, 1.214530e-01, 0.131174505, -0.127144501, 0.0237286594, -0.129713804, -0.138007894, 0.166811496, -0.118656904]> : tensor<10xf32>) : tensor<10xf32>
}











