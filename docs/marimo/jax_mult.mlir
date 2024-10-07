#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
builtin.module {
  func.func public @main(%arg0: tensor<2x3xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<3x4xf32> {mhlo.layout_mode = "default"}, %arg2: tensor<2x4xf32> {mhlo.layout_mode = "default", tf.aliasing_output = 0 : i32}) -> tensor<2x4xf32> {
    %0 = tensor.empty() : tensor<2x4xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>) outs(%1 : tensor<2x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<2x4xf32>
    return %2 : tensor<2x4xf32>
  }
}
