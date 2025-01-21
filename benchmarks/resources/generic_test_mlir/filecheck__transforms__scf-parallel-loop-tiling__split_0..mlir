"builtin.module"() ({
  "func.func"() <{function_type = (index, index, index, index, index, index, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> (), sym_name = "parallel_loop"}> ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: memref<?x?xf32>, %arg7: memref<?x?xf32>, %arg8: memref<?x?xf32>, %arg9: memref<?x?xf32>):
    "scf.parallel"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
    ^bb0(%arg10: index, %arg11: index):
      %0 = "memref.load"(%arg7, %arg10, %arg11) : (memref<?x?xf32>, index, index) -> f32
      %1 = "memref.load"(%arg8, %arg10, %arg11) : (memref<?x?xf32>, index, index) -> f32
      %2 = "arith.addf"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%2, %arg9, %arg10, %arg11) : (f32, memref<?x?xf32>, index, index) -> ()
      "scf.reduce"() : () -> ()
    }) : (index, index, index, index, index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
