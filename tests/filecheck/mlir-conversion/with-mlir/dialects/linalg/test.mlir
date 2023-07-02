#map = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @foo(%arg0: tensor<1x128xf32>, %arg1: tensor<128x256xf32>, %arg2: tensor<256xf32>) -> tensor<1x256xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x256xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x256xf32>) -> tensor<1x256xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<1x128xf32>, tensor<128x256xf32>) outs(%1 : tensor<1x256xf32>) -> tensor<1x256xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%2, %arg2 : tensor<1x256xf32>, tensor<256xf32>) outs(%0 : tensor<1x256xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.addf %in, %in_0 : f32
      linalg.yield %5 : f32
    } -> tensor<1x256xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<1x256xf32>) outs(%0 : tensor<1x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.cmpf ugt, %in, %cst : f32
      %6 = arith.select %5, %in, %cst : f32
      linalg.yield %6 : f32
    } -> tensor<1x256xf32>
    return %4 : tensor<1x256xf32>
  }
}
