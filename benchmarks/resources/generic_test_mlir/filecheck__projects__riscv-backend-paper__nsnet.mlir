"builtin.module"() ({
  "func.func"() <{function_type = (memref<1x161xf64>, memref<5x161xf64, strided<[161, 1]>>, memref<1x5xf64, strided<[40, 1]>>) -> (), sym_name = "main$async_dispatch_0_matmul_transpose_b_1x400x161_f64$xdsl_kernel1"}> ({
  ^bb0(%arg0: memref<1x161xf64>, %arg1: memref<5x161xf64, strided<[161, 1]>>, %arg2: memref<1x5xf64, strided<[40, 1]>>):
    "linalg.generic"(%arg0, %arg1, %arg2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg3: f64, %arg4: f64, %arg5: f64):
      %0 = "arith.mulf"(%arg3, %arg4) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
      %1 = "arith.addf"(%arg5, %0) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
      "linalg.yield"(%1) : (f64) -> ()
    }) : (memref<1x161xf64>, memref<5x161xf64, strided<[161, 1]>>, memref<1x5xf64, strided<[40, 1]>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
