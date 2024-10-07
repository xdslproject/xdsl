#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10) -> (d10, d0, d2 + d3, d4 + d5, d6 + d7, d8 + d9)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10) -> (d1, d0, d3, d5, d7, d9)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10) -> (d10, d1, d2, d4, d6, d8)>
#map5 = affine_map<(d0, d1, d2, d3) -> (-d0, -d1, -d2 + 2, -d3 + 2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
builtin.module {
  func.func public @main(%arg0: tensor<1x1x10x10xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<1x1x3x3xf32> {mhlo.layout_mode = "default"}, %arg2: tensor<1x1x8x8xf32> {mhlo.layout_mode = "default", tf.aliasing_output = 0 : i32}) -> tensor<1x1x8x8xf32> {
    %0 = call @_flip(%arg1) : (tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
    %1 = tensor.empty() : tensor<1x1x1x1x10x10xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1x10x10xf32>) outs(%1 : tensor<1x1x1x1x10x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1x1x1x10x10xf32>
    %3 = tensor.empty() : tensor<1x1x1x1x3x3xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%0 : tensor<1x1x3x3xf32>) outs(%3 : tensor<1x1x1x1x3x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1x1x1x3x3xf32>
    %5 = tensor.empty() : tensor<1x1x1x1x8x8xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x1x1x1x8x8xf32>) -> tensor<1x1x1x1x8x8xf32>
    %7 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "reduction", "parallel", "reduction", "parallel", "reduction", "parallel"]} ins(%2, %4 : tensor<1x1x1x1x10x10xf32>, tensor<1x1x1x1x3x3xf32>) outs(%6 : tensor<1x1x1x1x8x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %8 = arith.mulf %in, %in_0 : f32
      %9 = arith.addf %out, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<1x1x1x1x8x8xf32>
    %collapsed = tensor.collapse_shape %7 [[0], [1], [2, 3, 4], [5]] : tensor<1x1x1x1x8x8xf32> into tensor<1x1x8x8xf32>

    %output_1 = bufferization.materialize_in_destination %collapsed in %arg2 : (tensor<1x1x8x8xf32>, tensor<1x1x8x8xf32>) -> tensor<1x1x8x8xf32>

    return %output_1 : tensor<1x1x8x8xf32>
  }
  func.func private @_flip(%arg0: tensor<1x1x3x3xf32> {mhlo.layout_mode = "default"}) -> tensor<1x1x3x3xf32> {
    %0 = tensor.empty() : tensor<1x1x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1x3x3xf32>) outs(%0 : tensor<1x1x3x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1x3x3xf32>
    return %1 : tensor<1x1x3x3xf32>
  }
}
