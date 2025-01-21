"builtin.module"() ({
  %0:3 = "test.op"() : () -> (memref<f64>, memref<f64>, memref<f64>)
  %1:3 = "test.op"() : () -> (memref<2x3xf64>, memref<3x4xf64>, memref<2x4xf64>)
  %2:3 = "test.op"() : () -> (memref<4xf64>, memref<2xf64>, memref<3xf64>)
  "linalg.generic"(%0#0, %0#1, %0#2) <{doc = "documentation string", indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = [], library_call = "library call", operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg6: f64, %arg7: f64, %arg8: f64):
    %7 = "arith.mulf"(%arg6, %arg7) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %8 = "arith.addf"(%arg8, %7) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    "linalg.yield"(%8) : (f64) -> ()
  }) : (memref<f64>, memref<f64>, memref<f64>) -> ()
  "linalg.generic"(%1#0, %1#1, %1#2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg3: f64, %arg4: f64, %arg5: f64):
    %5 = "arith.mulf"(%arg3, %arg4) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %6 = "arith.addf"(%arg5, %5) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    "linalg.yield"(%6) : (f64) -> ()
  }) : (memref<2x3xf64>, memref<3x4xf64>, memref<2x4xf64>) -> ()
  "linalg.generic"(%2#0, %2#1, %2#2) <{indexing_maps = [affine_map<(d0, d1) -> (d0 + d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg0: f64, %arg1: f64, %arg2: f64):
    %3 = "arith.mulf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %4 = "arith.addf"(%arg2, %3) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    "linalg.yield"(%4) : (f64) -> ()
  }) : (memref<4xf64>, memref<2xf64>, memref<3xf64>) -> ()
}) : () -> ()
