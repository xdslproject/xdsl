"builtin.module"() ({
  %0:3 = "test.op"() : () -> (memref<f64>, memref<f64>, memref<f64>)
  %1:3 = "test.op"() : () -> (memref<2x3xf64>, memref<3x4xf64>, memref<2x4xf64>)
  %2:3 = "test.op"() : () -> (memref<4xf64>, memref<2xf64>, memref<3xf64>)
  "linalg.generic"(%0#0, %0#1, %0#2) <{indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = [], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg8: f64, %arg9: f64, %arg10: f64):
    %8 = "arith.mulf"(%arg8, %arg9) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %9 = "arith.addf"(%arg10, %8) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    "linalg.yield"(%9) : (f64) -> ()
  }) : (memref<f64>, memref<f64>, memref<f64>) -> ()
  "linalg.generic"(%1#0, %1#1, %1#2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg5: f64, %arg6: f64, %arg7: f64):
    %6 = "arith.mulf"(%arg5, %arg6) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %7 = "arith.addf"(%arg7, %6) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    "linalg.yield"(%7) : (f64) -> ()
  }) : (memref<2x3xf64>, memref<3x4xf64>, memref<2x4xf64>) -> ()
  "linalg.generic"(%2#0, %2#1, %2#2) <{indexing_maps = [affine_map<(d0, d1) -> (d0 + d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg2: f64, %arg3: f64, %arg4: f64):
    %4 = "arith.mulf"(%arg2, %arg3) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %5 = "arith.addf"(%arg4, %4) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    "linalg.yield"(%5) : (f64) -> ()
  }) : (memref<4xf64>, memref<2xf64>, memref<3xf64>) -> ()
  %3 = "arith.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
  "linalg.generic"(%3, %1#0) <{indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg0: f64, %arg1: f64):
    "linalg.yield"(%arg0) : (f64) -> ()
  }) : (f64, memref<2x3xf64>) -> ()
}) : () -> ()
