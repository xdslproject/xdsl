"builtin.module"() ({
  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
  %1 = "arith.constant"() <{value = 1000 : index}> : () -> index
  %2 = "arith.constant"() <{value = 3 : index}> : () -> index
  %3 = "arith.constant"() <{value = 1.020000e+01 : f32}> : () -> f32
  %4 = "arith.constant"() <{value = 1.810000e+01 : f32}> : () -> f32
  %5 = "scf.parallel"(%0, %1, %2, %3) <{operandSegmentSizes = array<i32: 1, 1, 1, 1>}> ({
  ^bb0(%arg0: index):
    "scf.reduce"(%4) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %6 = "arith.addf"(%arg1, %arg2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "scf.reduce.return"(%6) : (f32) -> ()
    }) : (f32) -> ()
  }) : (index, index, index, f32) -> f32
}) : () -> ()
