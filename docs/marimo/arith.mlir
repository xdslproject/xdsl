#map = affine_map<(d0, d1) -> (d0, d1)>
builtin.module {
  func.func public @main(%arg0: tensor<5x5xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<5x5xf32> {mhlo.layout_mode = "default"}, %arg2: tensor<5x5xf32> {mhlo.layout_mode = "default", tf.aliasing_output = 0 : i32}) -> tensor<5x5xf32> {
    %0 = tensor.empty() : tensor<5x5xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<5x5xf32>, tensor<5x5xf32>) outs(%0 : tensor<5x5xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %4 = arith.addf %in, %in_1 : f32
      linalg.yield %4 : f32
    } -> tensor<5x5xf32>
    %cst = arith.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<5x5xf32>
    %2 = tensor.empty() : tensor<5x5xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %cst_0 : tensor<5x5xf32>, tensor<5x5xf32>) outs(%2 : tensor<5x5xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %4 = arith.addf %in, %in_1 : f32
      linalg.yield %4 : f32
    } -> tensor<5x5xf32>

    %output = bufferization.materialize_in_destination %3 in %arg2 : (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>

    return %output : tensor<5x5xf32>
  }
}
