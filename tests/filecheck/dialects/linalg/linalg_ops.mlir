// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%0, %1 = "test.op"() : () -> (f32, memref<1x256xf32>)

linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : f32) outs(%1 : memref<1x256xf32>) {
^bb0(%arg3: f32, %arg4: f32):
    linalg.yield %arg3 : f32
}

linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"], doc="a_docstring", library_call="a_library_call"} ins(%0 : f32) outs(%1 : memref<1x256xf32>) {
^bb0(%arg3: f32, %arg4: f32):
    linalg.yield %arg3 : f32
}

linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : f32) outs(%1 : memref<1x256xf32>) attrs = {hello="world"} {
^bb0(%arg3: f32, %arg4: f32):
    linalg.yield %arg3 : f32
}


%in_t1, %in_t2, %out_t1 = "test.op"() : () -> (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>)
%in_m1, %in_m2, %out_m1 = "test.op"() : () -> (memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32>)

%sum = linalg.add ins(%in_t1, %in_t2 : tensor<4x16xf32>, tensor<4x16xf32>) outs(%out_t1 : tensor<4x16xf32>) -> tensor<4x16xf32>
linalg.add ins(%in_m1, %in_m2 : memref<4x16xf32>, memref<4x16xf32>) outs(%out_m1 : memref<4x16xf32>) -> ()


%2, %3 = "test.op"() : () -> (memref<64x9216xf32>, memref<9216x4096xf32>)
%4 = "test.op"() : () -> (memref<64x4096xf32>)
linalg.matmul {id} ins(%2, %3 : memref<64x9216xf32>, memref<9216x4096xf32>) outs(%4 : memref<64x4096xf32>)

// CHECK:        module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (f32, memref<1x256xf32>)
// CHECK-NEXT:    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : f32) outs(%1 : memref<1x256xf32>) {
// CHECK-NEXT:    ^0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:      linalg.yield %{{.*}} : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"], doc = "a_docstring", library_call = "a_library_call"} ins(%0 : f32) outs(%1 : memref<1x256xf32>) {
// CHECK-NEXT:    ^1(%arg3_1 : f32, %arg4_1 : f32):
// CHECK-NEXT:      linalg.yield %arg3_1 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : f32) outs(%1 : memref<1x256xf32>) attrs = {"hello" = "world"} {
// CHECK-NEXT:    ^{{.*}}(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:      linalg.yield %{{.*}} : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %in_t1, %in_t2, %out_t1 = "test.op"() : () -> (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>)
// CHECK-NEXT:    %in_m1, %in_m2, %out_m1 = "test.op"() : () -> (memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32>)
// CHECK-NEXT:    %sum = linalg.add ins(%in_t1, %in_t2 : tensor<4x16xf32>, tensor<4x16xf32>) outs(%out_t1 : tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:    linalg.add ins(%in_m1, %in_m2 : memref<4x16xf32>, memref<4x16xf32>) outs(%out_m1 : memref<4x16xf32>)
// CHECK-NEXT:    %2, %3 = "test.op"() : () -> (memref<64x9216xf32>, memref<9216x4096xf32>)
// CHECK-NEXT:    %4 = "test.op"() : () -> memref<64x4096xf32>
// CHECK-NEXT:    linalg.matmul {"id"} ins(%2, %3 : memref<64x9216xf32>, memref<9216x4096xf32>) outs(%4 : memref<64x4096xf32>)
// CHECK-NEXT:  }

// CHECK-GENERIC:       "linalg.generic"(%0, %1) <{"indexing_maps" = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], "iterator_types" = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:  ^0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:  }) : (f32, memref<1x256xf32>) -> ()
// CHECK-GENERIC-NEXT:  "linalg.generic"(%0, %1) <{"indexing_maps" = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], "iterator_types" = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], "doc" = "a_docstring", "library_call" = "a_library_call", "operandSegmentSizes" = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:  ^1(%arg3_1 : f32, %arg4_1 : f32):
// CHECK-GENERIC-NEXT:    "linalg.yield"(%arg3_1) : (f32) -> ()
// CHECK-GENERIC-NEXT:  }) : (f32, memref<1x256xf32>) -> ()
  
// CHECK-GENERIC:       "linalg.generic"(%0, %1) <{"indexing_maps" = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], "iterator_types" = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:  ^{{.*}}(%{{.*}}: f32, %{{.*}}: f32):
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:  }) {"hello" = "world"} : (f32, memref<1x256xf32>) -> ()

// CHECK-GENERIC-NEXT:    %in_t1, %in_t2, %out_t1 = "test.op"() : () -> (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>)
// CHECK-GENERIC-NEXT:    %in_m1, %in_m2, %out_m1 = "test.op"() : () -> (memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32>)
// CHECK-GENERIC-NEXT:    %sum = "linalg.add"(%in_t1, %in_t2, %out_t1) <{"operandSegmentSizes" = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^3(%2 : f32, %3 : f32, %4 : f32):
// CHECK-GENERIC-NEXT:      %5 = "arith.addf"(%2, %3) : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%5) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-GENERIC-NEXT:    "linalg.add"(%in_m1, %in_m2, %out_m1) <{"operandSegmentSizes" = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^4(%6 : f32, %7 : f32, %8 : f32):
// CHECK-GENERIC-NEXT:      %9 = "arith.addf"(%6, %7) : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%9) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32>) -> ()
