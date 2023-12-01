// RUN: XDSL_ROUNDTRIP

%0, %1 = "test.op"() : () -> (f32, memref<1x256xf32>)

"linalg.generic"(%0, %1) <{"indexing_maps" = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], "iterator_types" = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 1, 1>}> ({
^bb0(%arg3: f32, %arg4: f32):
    linalg.yield %arg3 : f32
}) : (f32, memref<1x256xf32>) -> ()

// CHECK:        module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (f32, memref<1x256xf32>)
// CHECK-NEXT:    "linalg.generic"(%0, %1) <{"indexing_maps" = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], "iterator_types" = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 1, 1>}> ({
// CHECK-NEXT:    ^0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:      linalg.yield %{{.*}} : f32
// CHECK-NEXT:    }) : (f32, memref<1x256xf32>) -> ()
// CHECK-NEXT:  }
