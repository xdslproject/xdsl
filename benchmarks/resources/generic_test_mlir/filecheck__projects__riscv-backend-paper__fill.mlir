"builtin.module"() ({
  "func.func"() <{function_type = (f64, memref<16x16xf64>) -> memref<16x16xf64>, sym_name = "fill", sym_visibility = "public"}> ({
  ^bb0(%arg0: f64, %arg1: memref<16x16xf64>):
    "linalg.generic"(%arg0, %arg1) <{indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg2: f64, %arg3: f64):
      "linalg.yield"(%arg2) : (f64) -> ()
    }) : (f64, memref<16x16xf64>) -> ()
    "func.return"(%arg1) : (memref<16x16xf64>) -> ()
  }) : () -> ()
}) : () -> ()
