// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

%0, %1 = "test.op"() : () -> (f32, memref<1x256xf32>)

"linalg.generic"(%0, %1) ({
^bb0(%arg3: f32, %arg4: f32):
    linalg.yield %arg3 : f32
}) {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>} : (f32, memref<1x256xf32>) -> ()


linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : f32) outs(%1 : memref<1x256xf32>) attrs = {hello="world"} {
^bb0(%arg3: f32, %arg4: f32):
    linalg.yield %arg3 : f32
}

%2, %3 = "test.op"() : () -> (tensor<2x3xf32>, tensor<2x3xf32>)

%sum = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %2 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%2 : tensor<2x3xf32>) {
^bb0(%in: f32, %in_0: f32, %out: f32):
    %acc = arith.addf %in, %in_0 : f32
    linalg.yield %acc : f32
} -> tensor<2x3xf32>

// CHECK-NEXT:  #map = affine_map<(d0, d1) -> ()>
// CHECK-NEXT:  #map1 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:  module {
// CHECK-NEXT:    %0:2 = "test.op"() : () -> (f32, memref<1x256xf32>)
// CHECK-NEXT:    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%0#0 : f32) outs(%0#1 : memref<1x256xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      linalg.yield %in : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%0#0 : f32) outs(%0#1 : memref<1x256xf32>) attrs =  {hello = "world"} {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      linalg.yield %in : f32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:    %1:2 = "test.op"() : () -> (tensor<2x3xf32>, tensor<2x3xf32>)
// CHECK-NEXT:    %2 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1#0, %1#0 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%1#0 : tensor<2x3xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK-NEXT:      %3 = arith.addf %in, %in_0 : f32
// CHECK-NEXT:      linalg.yield %4 : f32
// CHECK-NEXT:    } -> tensor<2x3xf32>
// CHECK-NEXT:  }
