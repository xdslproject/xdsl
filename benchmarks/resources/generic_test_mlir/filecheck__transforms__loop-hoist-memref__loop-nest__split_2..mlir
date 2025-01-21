"builtin.module"() ({
  "func.func"() <{arg_attrs = [{llvm.noalias}, {llvm.noalias}, {llvm.noalias}], function_type = (memref<8x8xf64>, memref<8x8xf64>, memref<8x8xf64>) -> memref<8x8xf64>, sym_name = "matmul", sym_visibility = "public"}> ({
  ^bb0(%arg0: memref<8x8xf64>, %arg1: memref<8x8xf64>, %arg2: memref<8x8xf64>):
    %0 = "arith.constant"() <{value = 0 : index}> : () -> index
    %1 = "arith.constant"() <{value = 8 : index}> : () -> index
    %2 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%0, %1, %2) ({
    ^bb0(%arg3: index):
      "scf.for"(%0, %1, %2) ({
      ^bb0(%arg4: index):
        "scf.for"(%0, %1, %2) ({
        ^bb0(%arg5: index):
          %3 = "memref.load"(%arg0, %arg3, %arg5) : (memref<8x8xf64>, index, index) -> f64
          %4 = "memref.load"(%arg1, %arg5, %arg4) : (memref<8x8xf64>, index, index) -> f64
          %5 = "memref.load"(%arg2, %arg3, %arg4) : (memref<8x8xf64>, index, index) -> f64
          %6 = "arith.mulf"(%3, %4) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
          %7 = "arith.addf"(%5, %6) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
          "memref.store"(%7, %arg2, %arg3, %arg4) : (f64, memref<8x8xf64>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"(%arg2) : (memref<8x8xf64>) -> ()
  }) : () -> ()
}) : () -> ()
