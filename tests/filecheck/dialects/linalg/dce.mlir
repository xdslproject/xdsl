// RUN: xdsl-opt %s -p dce | filecheck %s

%m = "test.op"() : () -> memref<4xf32>
linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%m : memref<4xf32>) outs(%m : memref<4xf32>) {
^bb0(%x: f32, %y: f32):
  %i = linalg.index 0 : index
  linalg.yield %x : f32
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %m = "test.op"() : () -> memref<4xf32>
// CHECK-NEXT:    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%m : memref<4xf32>) outs(%m : memref<4xf32>) {
// CHECK-NEXT:    ^{{.*}}(%x: f32, %y: f32):
// CHECK-NEXT:      linalg.yield %x : f32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
