"builtin.module"() ({
  "func.func"() <{function_type = (memref<128xi32>) -> i32, sym_name = "sum_vec", sym_visibility = "private"}> ({
  ^bb0(%arg6: memref<128xi32>):
    %5 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %6 = "affine.for"(%5) <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 1>, step = 1 : index, upperBoundMap = affine_map<() -> (256)>}> ({
    ^bb0(%arg7: index, %arg8: i32):
      %7 = "memref.load"(%arg6, %arg7) : (memref<128xi32>, index) -> i32
      %8 = "arith.addi"(%arg8, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "affine.yield"(%8) : (i32) -> ()
    }) : (i32) -> i32
    "func.return"(%6) : (i32) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (memref<256x256xf32>, memref<256x256xf32>, memref<256x256xf32>) -> memref<256x256xf32>, sym_name = "affine_mm", sym_visibility = "private"}> ({
  ^bb0(%arg0: memref<256x256xf32>, %arg1: memref<256x256xf32>, %arg2: memref<256x256xf32>):
    "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (256)>}> ({
    ^bb0(%arg3: index):
      "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (256)>}> ({
      ^bb0(%arg4: index):
        "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (256)>}> ({
        ^bb0(%arg5: index):
          %0 = "memref.load"(%arg0, %arg3, %arg5) : (memref<256x256xf32>, index, index) -> f32
          %1 = "memref.load"(%arg1, %arg5, %arg4) : (memref<256x256xf32>, index, index) -> f32
          %2 = "memref.load"(%arg2, %arg3, %arg4) : (memref<256x256xf32>, index, index) -> f32
          %3 = "arith.mulf"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %4 = "arith.addf"(%2, %3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%4, %arg2, %arg3, %arg4) : (f32, memref<256x256xf32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"(%arg2) : (memref<256x256xf32>) -> ()
  }) : () -> ()
}) : () -> ()
