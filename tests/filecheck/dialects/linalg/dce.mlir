// RUN: xdsl-opt %s -p dce | filecheck %s

%t, %m = "test.op"() {is_kept} : () -> (tensor<4xf32>, memref<4xf32>)
%0 = linalg.add ins(%t, %t : tensor<4xf32>, tensor<4xf32>) outs(%t : tensor<4xf32>) -> tensor<4xf32>
linalg.add ins(%m, %m : memref<4xf32>, memref<4xf32>) outs(%m : memref<4xf32>)
%1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%t : tensor<4xf32>) outs(%t : tensor<4xf32>) {
^bb0(%x: f32, %y: f32):
  %i = linalg.index 0 : index
  linalg.yield %x : f32
} -> tensor<4xf32>
linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%m : memref<4xf32>) outs(%m : memref<4xf32>) {
^bb0(%x: f32, %y: f32):
  %i = linalg.index 0 : index
  linalg.yield %x : f32
}
%2 = linalg.reduce ins(%t : tensor<4xf32>) outs(%t : tensor<4xf32>) dimensions = []
(%x : f32, %y : f32) {
  %s = arith.addf %x, %y : f32
  linalg.yield %s : f32
}
%3 = "test.op"() {is_kept} : () -> tensor<2x4xf32>
%4 = linalg.broadcast ins(%t : tensor<4xf32>) outs(%3 : tensor<2x4xf32>) dimensions = [0]

// CHECK:       builtin.module {
// CHECK-NEXT:    %t, %m = "test.op"() {is_kept} : () -> (tensor<4xf32>, memref<4xf32>)
// CHECK-NEXT:    linalg.add ins(%m, %m : memref<4xf32>, memref<4xf32>) outs(%m : memref<4xf32>)
// CHECK-NEXT:    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%m : memref<4xf32>) outs(%m : memref<4xf32>) {
// CHECK-NEXT:    ^{{.*}}(%x: f32, %y: f32):
// CHECK-NEXT:      linalg.yield %x : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %{{.*}} = "test.op"() {is_kept} : () -> tensor<2x4xf32>
// CHECK-NEXT:  }
