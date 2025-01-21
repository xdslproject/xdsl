"builtin.module"() ({
  "func.func"() <{function_type = (memref<8x16xf64>, memref<8x16xf64>, memref<8x16xf64>) -> memref<8x16xf64>, sym_name = "dsum", sym_visibility = "public"}> ({
  ^bb0(%arg4: memref<8x16xf64>, %arg5: memref<8x16xf64>, %arg6: memref<8x16xf64>):
    "linalg.generic"(%arg4, %arg5, %arg6) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg7: f64, %arg8: f64, %arg9: f64):
      %2 = "arith.addf"(%arg7, %arg8) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
      "linalg.yield"(%2) : (f64) -> ()
    }) : (memref<8x16xf64>, memref<8x16xf64>, memref<8x16xf64>) -> ()
    "func.return"(%arg6) : (memref<8x16xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (memref<16x16xf64>, memref<16x16xf64>) -> memref<16x16xf64>, sym_name = "relu", sym_visibility = "public"}> ({
  ^bb0(%arg0: memref<16x16xf64>, %arg1: memref<16x16xf64>):
    %0 = "arith.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
    "linalg.generic"(%arg0, %arg1) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg2: f64, %arg3: f64):
      %1 = "arith.maximumf"(%arg2, %0) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
      "linalg.yield"(%1) : (f64) -> ()
    }) : (memref<16x16xf64>, memref<16x16xf64>) -> ()
    "func.return"(%arg1) : (memref<16x16xf64>) -> ()
  }) : () -> ()
}) : () -> ()
