"builtin.module"() ({
  "func.func"() <{function_type = (memref<8xf64>, memref<8xf64>, memref<f64>) -> memref<f64>, sym_name = "ddot", sym_visibility = "public"}> ({
  ^bb0(%arg0: memref<8xf64>, %arg1: memref<8xf64>, %arg2: memref<f64>):
    %0 = "arith.constant"() <{value = 0 : index}> : () -> index
    %1 = "arith.constant"() <{value = 8 : index}> : () -> index
    %2 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%0, %1, %2) ({
    ^bb0(%arg3: index):
      %3 = "memref.load"(%arg0, %arg3) : (memref<8xf64>, index) -> f64
      %4 = "memref.load"(%arg1, %arg3) : (memref<8xf64>, index) -> f64
      %5 = "memref.load"(%arg2) : (memref<f64>) -> f64
      %6 = "arith.mulf"(%3, %4) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
      %7 = "arith.addf"(%5, %6) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
      "memref.store"(%7, %arg2) : (f64, memref<f64>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"(%arg2) : (memref<f64>) -> ()
  }) : () -> ()
}) : () -> ()
