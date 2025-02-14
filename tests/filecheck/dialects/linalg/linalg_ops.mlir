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


%t1, %t2, %t3 = "test.op"() : () -> (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>)
%m1, %m2, %m3 = "test.op"() : () -> (memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32>)

%sum = linalg.add ins(%t1, %t2 : tensor<4x16xf32>, tensor<4x16xf32>) outs(%t3 : tensor<4x16xf32>) -> tensor<4x16xf32>
linalg.add ins(%m1, %m2 : memref<4x16xf32>, memref<4x16xf32>) outs(%m3 : memref<4x16xf32>) -> ()

%mul = linalg.mul ins(%t1, %t2 : tensor<4x16xf32>, tensor<4x16xf32>) outs(%t3 : tensor<4x16xf32>) -> tensor<4x16xf32>
linalg.mul ins(%m1, %m2 : memref<4x16xf32>, memref<4x16xf32>) outs(%m3 : memref<4x16xf32>)

%2, %3 = "test.op"() : () -> (memref<64x9216xf32>, memref<9216x4096xf32>)
%4 = "test.op"() : () -> (memref<64x4096xf32>)
linalg.matmul {id} ins(%2, %3 : memref<64x9216xf32>, memref<9216x4096xf32>) outs(%4 : memref<64x4096xf32>)

%i2, %i3 = "test.op"() : () -> (memref<64x9216xi32>, memref<9216x4096xi32>)
%i4 = "test.op"() : () -> (memref<64x4096xi32>)
linalg.matmul {id} ins(%i2, %i3 : memref<64x9216xi32>, memref<9216x4096xi32>) outs(%i4 : memref<64x4096xi32>)

%fill = linalg.fill ins(%0 : f32) outs(%t3 : tensor<4x16xf32>) -> tensor<4x16xf32>
linalg.fill ins(%0 : f32) outs(%m3 : memref<4x16xf32>)

%5, %6 = "test.op"() : () -> (tensor<64x9216xi8>, tensor<9216x4096xi8>)
%7 = arith.constant 0 : i32
%8 = arith.constant 0 : i32
%9 = "test.op"() : () -> (tensor<64x4096xi32>)

linalg.quantized_matmul ins(%5, %6, %7, %8 : tensor<64x9216xi8>, tensor<9216x4096xi8>, i32, i32) outs(%9 : tensor<64x4096xi32>) -> tensor<64x4096xi32>


// CHECK:        module {
// CHECK-NEXT:    %{{.*}} %{{.*}} = "test.op"() : () -> (f32, memref<1x256xf32>)
// CHECK-NEXT:    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%{{.*}} : f32) outs(%{{.*}} : memref<1x256xf32>) {
// CHECK-NEXT:    ^0(%{{.*}} f32, %{{.*}} f32):
// CHECK-NEXT:      linalg.yield %{{.*}} : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"], doc = "a_docstring", library_call = "a_library_call"} ins(%{{.*}} : f32) outs(%{{.*}} : memref<1x256xf32>) {
// CHECK-NEXT:    ^1(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-NEXT:      linalg.yield %{{.*}} : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%{{.*}} : f32) outs(%{{.*}} : memref<1x256xf32>) attrs = {hello = "world"} {
// CHECK-NEXT:    ^{{.*}}(%{{.*}} f32, %{{.*}} f32):
// CHECK-NEXT:      linalg.yield %{{.*}} : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %{{.*}} %{{.*}} %{{.*}} = "test.op"() : () -> (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>)
// CHECK-NEXT:    %{{.*}} %{{.*}} %{{.*}} = "test.op"() : () -> (memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32>)
// CHECK-NEXT:    %{{.*}} = linalg.add ins(%{{.*}} %{{.*}} : tensor<4x16xf32>, tensor<4x16xf32>) outs(%{{.*}} : tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:    linalg.add ins(%{{.*}} %{{.*}} : memref<4x16xf32>, memref<4x16xf32>) outs(%{{.*}} : memref<4x16xf32>)
// CHECK-NEXT:    %{{.*}} = linalg.mul ins(%{{.*}} %{{.*}} : tensor<4x16xf32>, tensor<4x16xf32>) outs(%{{.*}} : tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:    linalg.mul ins(%{{.*}} %{{.*}} : memref<4x16xf32>, memref<4x16xf32>) outs(%{{.*}} : memref<4x16xf32>)
// CHECK-NEXT:    %{{.*}} %{{.*}} = "test.op"() : () -> (memref<64x9216xf32>, memref<9216x4096xf32>)
// CHECK-NEXT:    %{{.*}} = "test.op"() : () -> memref<64x4096xf32>
// CHECK-NEXT:    linalg.matmul {id} ins(%{{.*}} %{{.*}} : memref<64x9216xf32>, memref<9216x4096xf32>) outs(%{{.*}} : memref<64x4096xf32>)
// CHECK-NEXT:    %{{.*}} %{{.*}} = "test.op"() : () -> (memref<64x9216xi32>, memref<9216x4096xi32>)
// CHECK-NEXT:    %{{.*}} = "test.op"() : () -> memref<64x4096xi32>
// CHECK-NEXT:    linalg.matmul {id} ins(%{{.*}} %{{.*}} : memref<64x9216xi32>, memref<9216x4096xi32>) outs(%{{.*}} : memref<64x4096xi32>)
// CHECK-NEXT:    %{{.*}} = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:    linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : memref<4x16xf32>)
// CHECK-NEXT:    %5, %6 = "test.op"() : () -> (tensor<64x9216xi8>, tensor<9216x4096xi8>)
// CHECK-NEXT:    %7 = arith.constant 0 : i32
// CHECK-NEXT:    %8 = arith.constant 0 : i32
// CHECK-NEXT:    %9 = "test.op"() : () -> tensor<64x4096xi32>
// CHECK-NEXT:    linalg.quantized_matmul ins(%5, %6, %7, %8 : tensor<64x9216xi8>, tensor<9216x4096xi8>, i32, i32) outs(%9 : tensor<64x4096xi32>) -> tensor<64x4096xi32>
// CHECK-NEXT:    }

// CHECK-GENERIC:       "linalg.generic"(%{{.*}} %{{.*}} <{indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:  ^0(%{{.*}} f32, %{{.*}} f32):
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}} : (f32) -> ()
// CHECK-GENERIC-NEXT:  }) : (f32, memref<1x256xf32>) -> ()
// CHECK-GENERIC-NEXT:  "linalg.generic"(%{{.*}} %{{.*}} <{indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], doc = "a_docstring", library_call = "a_library_call", operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:  ^1(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:    "linalg.yield"(%{{.*}} : (f32) -> ()
// CHECK-GENERIC-NEXT:  }) : (f32, memref<1x256xf32>) -> ()

// CHECK-GENERIC:       "linalg.generic"(%{{.*}} %{{.*}} <{indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:  ^{{.*}}(%{{.*}} f32, %{{.*}} f32):
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}} : (f32) -> ()
// CHECK-GENERIC-NEXT:  }) {hello = "world"} : (f32, memref<1x256xf32>) -> ()

// CHECK-GENERIC-NEXT:    %{{.*}} %{{.*}} %{{.*}} = "test.op"() : () -> (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>)
// CHECK-GENERIC-NEXT:    %{{.*}} %{{.*}} %{{.*}} = "test.op"() : () -> (memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32>)

// CHECK-GENERIC-NEXT:    %{{.*}} = "linalg.add"(%{{.*}} %{{.*}} %{{.*}} <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^3(%{{.*}} : f32, %{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addf"(%{{.*}} %{{.*}} : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}} : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>

// CHECK-GENERIC-NEXT:    "linalg.add"(%{{.*}} %{{.*}} %{{.*}} <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^4(%{{.*}} : f32, %{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addf"(%{{.*}} %{{.*}} : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}} : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32>) -> ()

// CHECK-GENERIC-NEXT:    %{{.*}} = "linalg.mul"(%{{.*}} %{{.*}} %{{.*}} <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^5(%{{.*}} : f32, %{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.mulf"(%{{.*}} %{{.*}} : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}} : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>

// CHECK-GENERIC-NEXT:    "linalg.mul"(%{{.*}} %{{.*}} %{{.*}} <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^6(%{{.*}} : f32, %{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.mulf"(%{{.*}} %{{.*}} : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}} : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32>) -> ()

// CHECK-GENERIC-NEXT:    %{{.*}} %{{.*}} = "test.op"() : () -> (memref<64x9216xf32>, memref<9216x4096xf32>)
// CHECK-GENERIC-NEXT:    %{{.*}} = "test.op"() : () -> memref<64x4096xf32>

// CHECK-GENERIC-NEXT:    "linalg.matmul"(%{{.*}} %{{.*}} %{{.*}} <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^7(%{{.*}} : f32, %{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.mulf"(%{{.*}}, %{{.*}} : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}} : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}} : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) {id, linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]} : (memref<64x9216xf32>, memref<9216x4096xf32>, memref<64x4096xf32>) -> ()

// CHECK-GENERIC-NEXT:    %{{.*}} %{{.*}} = "test.op"() : () -> (memref<64x9216xi32>, memref<9216x4096xi32>)
// CHECK-GENERIC-NEXT:    %{{.*}} = "test.op"() : () -> memref<64x4096xi32>

// CHECK-GENERIC-NEXT:    "linalg.matmul"(%{{.*}} %{{.*}} %{{.*}} <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^8(%{{.*}} : i32, %{{.*}} : i32, %{{.*}} : i32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.muli"(%{{.*}}, %{{.*}} : (i32, i32) -> i32
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addi"(%{{.*}}, %{{.*}} : (i32, i32) -> i32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}} : (i32) -> ()
// CHECK-GENERIC-NEXT:    }) {id, linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]} : (memref<64x9216xi32>, memref<9216x4096xi32>, memref<64x4096xi32>) -> ()

// CHECK-GENERIC-NEXT:    %{{.*}} = "linalg.fill"(%{{.*}}, %{{.*}} <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^9(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}} : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (f32, tensor<4x16xf32>) -> tensor<4x16xf32>

// CHECK-GENERIC-NEXT:    "linalg.fill"(%{{.*}}, %{{.*}} <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^10(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}} : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (f32, memref<4x16xf32>) -> ()

// CHECK-GENERIC-NEXT:     %{{.*}}, %{{.*}} = "test.op"() : () -> (tensor<64x9216xi8>, tensor<9216x4096xi8>)
// CHECK-GENERIC-NEXT:     %{{.*}} = "arith.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-GENERIC-NEXT:     %{{.*}} = "arith.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-GENERIC-NEXT:     %{{.*}} = "test.op"() : () -> tensor<64x4096xi32>

// CHECK-GENERIC-NEXT:    %{{.*}} = "linalg.quantized_matmul"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 4, 1>}> ({
// CHECK-GENERIC-NEXT:       ^11(%41 : i8, %42 : i8, %43 : i32, %44 : i32, %45 : i32):
// CHECK-GENERIC-NEXT:         %46 = "arith.extsi"(%41) : (i8) -> i32
// CHECK-GENERIC-NEXT:         %47 = "arith.subi"(%46, %43) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
// CHECK-GENERIC-NEXT:         %48 = "arith.extsi"(%42) : (i8) -> i32
// CHECK-GENERIC-NEXT:         %49 = "arith.subi"(%48, %44) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
// CHECK-GENERIC-NEXT:         %50 = "arith.muli"(%47, %49) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
// CHECK-GENERIC-NEXT:         %51 = "arith.addi"(%45, %50) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
//  CHECK-GENERIC-NEXT:        "linalg.yield"(%51) : (i32) -> ()
// CHECK-GENERIC-NEXT:       }) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>]} : (tensor<64x9216xi8>, tensor<9216x4096xi8>, i32, i32, tensor<64x4096xi32>) -> tensor<64x4096xi32>
