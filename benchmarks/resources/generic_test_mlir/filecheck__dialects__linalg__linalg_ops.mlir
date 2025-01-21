"builtin.module"() ({
  %0:2 = "test.op"() : () -> (f32, memref<1x256xf32>)
  "linalg.generic"(%0#0, %0#1) <{indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg28: f32, %arg29: f32):
    "linalg.yield"(%arg28) : (f32) -> ()
  }) : (f32, memref<1x256xf32>) -> ()
  "linalg.generic"(%0#0, %0#1) <{doc = "a_docstring", indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], library_call = "a_library_call", operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg26: f32, %arg27: f32):
    "linalg.yield"(%arg26) : (f32) -> ()
  }) : (f32, memref<1x256xf32>) -> ()
  "linalg.generic"(%0#0, %0#1) <{indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg24: f32, %arg25: f32):
    "linalg.yield"(%arg24) : (f32) -> ()
  }) {hello = "world"} : (f32, memref<1x256xf32>) -> ()
  %1:3 = "test.op"() : () -> (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>)
  %2:3 = "test.op"() : () -> (memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32>)
  %3 = "linalg.add"(%1#0, %1#1, %1#2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg21: f32, %arg22: f32, %arg23: f32):
    %24 = "arith.addf"(%arg21, %arg22) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%24) : (f32) -> ()
  }) : (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
  "linalg.add"(%2#0, %2#1, %2#2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg18: f32, %arg19: f32, %arg20: f32):
    %23 = "arith.addf"(%arg18, %arg19) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%23) : (f32) -> ()
  }) : (memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32>) -> ()
  %4 = "linalg.mul"(%1#0, %1#1, %1#2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg15: f32, %arg16: f32, %arg17: f32):
    %22 = "arith.mulf"(%arg15, %arg16) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%22) : (f32) -> ()
  }) : (tensor<4x16xf32>, tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
  "linalg.mul"(%2#0, %2#1, %2#2) <{operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg12: f32, %arg13: f32, %arg14: f32):
    %21 = "arith.mulf"(%arg12, %arg13) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%21) : (f32) -> ()
  }) : (memref<4x16xf32>, memref<4x16xf32>, memref<4x16xf32>) -> ()
  %5:2 = "test.op"() : () -> (memref<64x9216xf32>, memref<9216x4096xf32>)
  %6 = "test.op"() : () -> memref<64x4096xf32>
  "linalg.matmul"(%5#0, %5#1, %6) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
    %19 = "arith.mulf"(%arg9, %arg10) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %20 = "arith.addf"(%arg11, %19) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%20) : (f32) -> ()
  }) {id} : (memref<64x9216xf32>, memref<9216x4096xf32>, memref<64x4096xf32>) -> ()
  %7 = "linalg.fill"(%0#0, %1#2) <{operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg7: f32, %arg8: f32):
    "linalg.yield"(%arg7) : (f32) -> ()
  }) : (f32, tensor<4x16xf32>) -> tensor<4x16xf32>
  "linalg.fill"(%0#0, %2#2) <{operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg5: f32, %arg6: f32):
    "linalg.yield"(%arg5) : (f32) -> ()
  }) : (f32, memref<4x16xf32>) -> ()
  %8:2 = "test.op"() : () -> (tensor<64x9216xi8>, tensor<9216x4096xi8>)
  %9 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  %10 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  %11 = "test.op"() : () -> tensor<64x4096xi32>
  %12 = "linalg.quantized_matmul"(%8#0, %8#1, %9, %10, %11) <{operandSegmentSizes = array<i32: 4, 1>}> ({
  ^bb0(%arg0: i8, %arg1: i8, %arg2: i32, %arg3: i32, %arg4: i32):
    %13 = "arith.extsi"(%arg0) : (i8) -> i32
    %14 = "arith.subi"(%13, %arg2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %15 = "arith.extsi"(%arg1) : (i8) -> i32
    %16 = "arith.subi"(%15, %arg3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %17 = "arith.muli"(%14, %16) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %18 = "arith.addi"(%arg4, %17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "linalg.yield"(%18) : (i32) -> ()
  }) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>]} : (tensor<64x9216xi8>, tensor<9216x4096xi8>, i32, i32, tensor<64x4096xi32>) -> tensor<64x4096xi32>
}) : () -> ()
