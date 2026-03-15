// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%0, %1 = "test.op"() : () -> (f32, memref<1x256xf32>)

linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : f32) outs(%1 : memref<1x256xf32>) {
^bb0(%arg3: f32, %arg4: f32):
    linalg.yield %arg3 : f32
}

linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : f32) outs(%1 : memref<1x256xf32>) {
^bb0(%arg3: f32, %arg4: f32):
    %index = linalg.index 0 : index
    %cast_index = arith.index_cast %index : index to i64
    %convert_index = arith.sitofp %cast_index : i64 to f32
    %sum = arith.addf %arg3, %convert_index : f32
    linalg.yield %sum : f32
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

 %copy = linalg.copy {id} ins(%t1 : tensor<4x16xf32>) outs(%t3 : tensor<4x16xf32>) -> tensor<4x16xf32>
 linalg.copy {id} ins(%m1 : memref<4x16xf32>) outs(%m3 : memref<4x16xf32>)

 %exp = linalg.exp ins(%t2: tensor<4x16xf32>) outs(%t3: tensor<4x16xf32>) -> tensor<4x16xf32>
 %log = linalg.log ins(%t2: tensor<4x16xf32>) outs(%t3: tensor<4x16xf32>) -> tensor<4x16xf32>
 %sqrt = linalg.sqrt ins(%t2: tensor<4x16xf32>) outs(%t3: tensor<4x16xf32>) -> tensor<4x16xf32>

 %5, %6 = "test.op"() : () -> (tensor<64x9216xi8>, tensor<9216x4096xi8>)
 %7 = arith.constant 0 : i32
 %8 = arith.constant 0 : i32
 %9 = "test.op"() : () -> (tensor<64x4096xi32>)

 linalg.quantized_matmul ins(%5, %6, %7, %8 : tensor<64x9216xi8>, tensor<9216x4096xi8>, i32, i32) outs(%9 : tensor<64x4096xi32>) -> tensor<64x4096xi32>

 %b1 = "test.op"() : () -> tensor<4x16xi1>
 %10 = linalg.select ins(%b1, %t1, %t2 : tensor<4x16xi1>, tensor<4x16xf32>, tensor<4x16xf32>) outs(%t3 : tensor<4x16xf32>) -> tensor<4x16xf32>
 "test.op"(%10) : (tensor<4x16xf32>) -> ()

 %11 = linalg.max ins(%t1, %t2 : tensor<4x16xf32>, tensor<4x16xf32>) outs(%t1 : tensor<4x16xf32>) -> tensor<4x16xf32>
 %12 = linalg.min ins(%t1, %t2 : tensor<4x16xf32>, tensor<4x16xf32>) outs(%t1 : tensor<4x16xf32>) -> tensor<4x16xf32>

 %13, %14 = "test.op"() : () -> (memref<64x10xi32>, memref<i32>)
 linalg.reduce ins(%13:memref<64x10xi32>) outs(%14:memref<i32>) dimensions = [0, 1]
 (%15 : i32, %16 : i32) {
     %17 = arith.addi %15, %16 : i32
     linalg.yield %17 : i32
 }

 %18, %19 = "test.op"() : () -> (tensor<12x20xf32>, tensor<20xf32>)
 linalg.reduce ins(%18:tensor<12x20xf32>) outs(%19:tensor<20xf32>) dimensions = [0]
 (%20 : f32, %21 : f32) {
     %22 = arith.addf %20, %21 : f32
     linalg.yield %22 : f32
 }


// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (f32, memref<1x256xf32>)
// CHECK-NEXT:    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%{{.*}} : f32) outs(%{{.*}} : memref<1x256xf32>) {
// CHECK-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-NEXT:      linalg.yield %{{.*}} : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%{{.*}} : f32) outs(%{{.*}} : memref<1x256xf32>) {
// CHECK-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-NEXT:      %{{.*}} = linalg.index 0 : index
// CHECK-NEXT:      %{{.*}} = arith.index_cast %{{.*}} : index to i64
// CHECK-NEXT:      %{{.*}} = arith.sitofp %{{.*}} : i64 to f32
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:      linalg.yield %{{.*}} : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"], doc = "a_docstring", library_call = "a_library_call"} ins(%{{.*}} : f32) outs(%{{.*}} : memref<1x256xf32>) {
// CHECK-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-NEXT:      linalg.yield %{{.*}} : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%{{.*}} : f32) outs(%{{.*}} : memref<1x256xf32>) attrs =  {hello = "world"} {
// CHECK-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-NEXT:      linalg.yield %{{.*}} : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>)
// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32>)
// CHECK-NEXT:    %{{.*}} = linalg.add ins(%{{.*}}, %{{.*}} : tensor<4x16xf32>, tensor<4x16xf32>) outs(%{{.*}} : tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:    linalg.add ins(%{{.*}}, %{{.*}} : memref<4x16xf32>, memref<4x16xf32>) outs(%{{.*}} : memref<4x16xf32>)
// CHECK-NEXT:    %{{.*}} = linalg.mul ins(%{{.*}}, %{{.*}} : tensor<4x16xf32>, tensor<4x16xf32>) outs(%{{.*}} : tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:    linalg.mul ins(%{{.*}}, %{{.*}} : memref<4x16xf32>, memref<4x16xf32>) outs(%{{.*}} : memref<4x16xf32>)
// CHECK-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<64x9216xf32>, memref<9216x4096xf32>)
// CHECK-NEXT:    %{{.*}} = "test.op"() : () -> memref<64x4096xf32>
// CHECK-NEXT:    linalg.matmul {id} ins(%{{.*}}, %{{.*}} : memref<64x9216xf32>, memref<9216x4096xf32>) outs(%{{.*}} : memref<64x4096xf32>)
// CHECK-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<64x9216xi32>, memref<9216x4096xi32>)
// CHECK-NEXT:    %{{.*}} = "test.op"() : () -> memref<64x4096xi32>
// CHECK-NEXT:    linalg.matmul {id} ins(%{{.*}}, %{{.*}} : memref<64x9216xi32>, memref<9216x4096xi32>) outs(%{{.*}} : memref<64x4096xi32>)
// CHECK-NEXT:    %{{.*}} = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:    linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : memref<4x16xf32>)
// CHECK-NEXT:    %{{.*}} = linalg.copy {id} ins(%{{.*}} : tensor<4x16xf32>) outs(%{{.*}} : tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:    linalg.copy {id} ins(%{{.*}} : memref<4x16xf32>) outs(%{{.*}} : memref<4x16xf32>)
// CHECK-NEXT:    %{{.*}} = linalg.exp ins(%{{.*}} : tensor<4x16xf32>) outs(%{{.*}} : tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:    %{{.*}} = linalg.log ins(%{{.*}} : tensor<4x16xf32>) outs(%{{.*}} : tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:    %{{.*}} = linalg.sqrt ins(%{{.*}} : tensor<4x16xf32>) outs(%{{.*}} : tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (tensor<64x9216xi8>, tensor<9216x4096xi8>)
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:    %{{.*}} = "test.op"() : () -> tensor<64x4096xi32>
// CHECK-NEXT:    %{{.*}} = linalg.quantized_matmul ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x9216xi8>, tensor<9216x4096xi8>, i32, i32) outs(%{{.*}} : tensor<64x4096xi32>) -> tensor<64x4096xi32>
// CHECK-NEXT:    %{{.*}} = "test.op"() : () -> tensor<4x16xi1>
// CHECK-NEXT:    %{{.*}} = linalg.select ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<4x16xi1>, tensor<4x16xf32>, tensor<4x16xf32>) outs(%{{.*}} : tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:    "test.op"(%{{.*}}) : (tensor<4x16xf32>) -> ()
// CHECK-NEXT:    %{{.*}} = linalg.max ins(%{{.*}}, %{{.*}} : tensor<4x16xf32>, tensor<4x16xf32>) outs(%{{.*}} : tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:    %{{.*}} = linalg.min ins(%{{.*}}, %{{.*}} : tensor<4x16xf32>, tensor<4x16xf32>) outs(%{{.*}} : tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<64x10xi32>, memref<i32>)
// CHECK-NEXT:    linalg.reduce ins(%{{.*}}memref<64x10xi32>) outs(%{{.*}}memref<i32>) dimensions = [0, 1]
// CHECK-NEXT:    (%{{.*}} : i32, %{{.*}} : i32) {
// CHECK-NEXT:      %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:      linalg.yield %{{.*}} : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (tensor<12x20xf32>, tensor<20xf32>)
// CHECK-NEXT:    %{{.*}} = linalg.reduce ins(%{{.*}}tensor<12x20xf32>) outs(%{{.*}}tensor<20xf32>) dimensions = [0]
// CHECK-NEXT:    (%{{.*}} : f32, %{{.*}} : f32) {
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:      linalg.yield %{{.*}} : f32
// CHECK-NEXT:    }
// CHECK-NEXT:  }


// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (f32, memref<1x256xf32>)
// CHECK-GENERIC-NEXT:    "linalg.generic"(%{{.*}}, %{{.*}}) <{indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (f32, memref<1x256xf32>) -> ()
// CHECK-GENERIC-NEXT:    "linalg.generic"(%{{.*}}, %{{.*}}) <{indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "linalg.index"() <{dim = 0 : i64}> : () -> index
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.index_cast"(%{{.*}}) : (index) -> i64
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.sitofp"(%{{.*}}) : (i64) -> f32
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (f32, memref<1x256xf32>) -> ()
// CHECK-GENERIC-NEXT:    "linalg.generic"(%{{.*}}, %{{.*}}) <{indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], doc = "a_docstring", library_call = "a_library_call", operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (f32, memref<1x256xf32>) -> ()
// CHECK-GENERIC-NEXT:    "linalg.generic"(%{{.*}}, %{{.*}}) <{indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) {hello = "world"} : (f32, memref<1x256xf32>) -> ()
// CHECK-GENERIC-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>)
// CHECK-GENERIC-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32>)
// CHECK-GENERIC-NEXT:    %{{.*}} = "linalg.add"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-GENERIC-NEXT:    "linalg.add"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32>) -> ()
// CHECK-GENERIC-NEXT:    %{{.*}} = "linalg.mul"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.mulf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-GENERIC-NEXT:    "linalg.mul"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.mulf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32>) -> ()
// CHECK-GENERIC-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<64x9216xf32>, memref<9216x4096xf32>)
// CHECK-GENERIC-NEXT:    %{{.*}} = "test.op"() : () -> memref<64x4096xf32>
// CHECK-GENERIC-NEXT:    "linalg.matmul"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.mulf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) {id} : (memref<64x9216xf32>, memref<9216x4096xf32>, memref<64x4096xf32>) -> ()
// CHECK-GENERIC-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<64x9216xi32>, memref<9216x4096xi32>)
// CHECK-GENERIC-NEXT:    %{{.*}} = "test.op"() : () -> memref<64x4096xi32>
// CHECK-GENERIC-NEXT:    "linalg.matmul"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : i32, %{{.*}} : i32, %{{.*}} : i32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.muli"(%{{.*}}, %{{.*}}) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addi"(%{{.*}}, %{{.*}}) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (i32) -> ()
// CHECK-GENERIC-NEXT:    }) {id} : (memref<64x9216xi32>, memref<9216x4096xi32>, memref<64x4096xi32>) -> ()
// CHECK-GENERIC-NEXT:    %{{.*}} = "linalg.fill"(%{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (f32, tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-GENERIC-NEXT:    "linalg.fill"(%{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (f32, memref<4x16xf32>) -> ()
// CHECK-GENERIC-NEXT:    %{{.*}} = "linalg.copy"(%{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) {id} : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-GENERIC-NEXT:    "linalg.copy"(%{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) {id} : (memref<4x16xf32>, memref<4x16xf32>) -> ()
// CHECK-GENERIC-NEXT:    %{{.*}} = "linalg.exp"(%{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "math.exp"(%{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-GENERIC-NEXT:    %{{.*}} = "linalg.log"(%{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "math.log"(%{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-GENERIC-NEXT:    %{{.*}} = "linalg.sqrt"(%{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "math.sqrt"(%{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-GENERIC-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (tensor<64x9216xi8>, tensor<9216x4096xi8>)
// CHECK-GENERIC-NEXT:    %{{.*}} = "arith.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-GENERIC-NEXT:    %{{.*}} = "arith.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-GENERIC-NEXT:    %{{.*}} = "test.op"() : () -> tensor<64x4096xi32>
// CHECK-GENERIC-NEXT:    %{{.*}} = "linalg.quantized_matmul"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 4, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : i8, %{{.*}} : i8, %{{.*}} : i32, %{{.*}} : i32, %{{.*}} : i32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.extsi"(%{{.*}}) : (i8) -> i32
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.subi"(%{{.*}}, %{{.*}}) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.extsi"(%{{.*}}) : (i8) -> i32
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.subi"(%{{.*}}, %{{.*}}) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.muli"(%{{.*}}, %{{.*}}) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addi"(%{{.*}}, %{{.*}}) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (i32) -> ()
// CHECK-GENERIC-NEXT:    }) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>]} : (tensor<64x9216xi8>, tensor<9216x4096xi8>, i32, i32, tensor<64x4096xi32>) -> tensor<64x4096xi32>
// CHECK-GENERIC-NEXT:    %{{.*}} = "test.op"() : () -> tensor<4x16xi1>
// CHECK-GENERIC-NEXT:    %{{.*}} = "linalg.select"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 3, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : i1, %{{.*}} : f32, %{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.select"(%{{.*}}, %{{.*}}, %{{.*}}) : (i1, f32, f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (tensor<4x16xi1>, tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-GENERIC-NEXT:    "test.op"(%{{.*}}) : (tensor<4x16xf32>) -> ()
// CHECK-GENERIC-NEXT:    %{{.*}} = "linalg.max"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.maximumf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-GENERIC-NEXT:    %{{.*}} = "linalg.min"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.minimumf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK-GENERIC-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<64x10xi32>, memref<i32>)
// CHECK-GENERIC-NEXT:    "linalg.reduce"(%{{.*}}, %{{.*}}) <{dimensions = array<i64: 0, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : i32, %{{.*}} : i32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addi"(%{{.*}}, %{{.*}}) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (i32) -> ()
// CHECK-GENERIC-NEXT:    }) : (memref<64x10xi32>, memref<i32>) -> ()
// CHECK-GENERIC-NEXT:    %{{.*}}, %{{.*}} = "test.op"() : () -> (tensor<12x20xf32>, tensor<20xf32>)
// CHECK-GENERIC-NEXT:    %{{.*}} = "linalg.reduce"(%{{.*}}, %{{.*}}) <{dimensions = array<i64: 0>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (tensor<12x20xf32>, tensor<20xf32>) -> tensor<20xf32>
// CHECK-GENERIC-NEXT:  }) : () -> ()
