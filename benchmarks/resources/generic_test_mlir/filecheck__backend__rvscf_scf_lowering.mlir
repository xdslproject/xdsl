"builtin.module"() ({
  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
  %1 = "arith.constant"() <{value = 10 : index}> : () -> index
  %2 = "arith.constant"() <{value = 1 : index}> : () -> index
  %3 = "arith.constant"() <{value = 42 : index}> : () -> index
  %4 = "arith.constant"() <{value = 2.100000e+01 : f32}> : () -> f32
  %5 = "arith.constant"() <{value = 4.200000e+01 : f64}> : () -> f64
  %6:3 = "scf.for"(%0, %1, %2, %3, %4, %5) ({
  ^bb0(%arg0: index, %arg1: index, %arg2: f32, %arg3: f64):
    %7 = "arith.addi"(%arg0, %arg1) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    "scf.yield"(%7, %arg2, %arg3) : (index, f32, f64) -> ()
  }) : (index, index, index, index, f32, f64) -> (index, f32, f64)
}) : () -> ()
