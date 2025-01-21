"builtin.module"() ({
  %0 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
  %1 = "arith.constant"() <{value = 1.000000e+00 : f64}> : () -> f64
  %2 = "arith.negf"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
  %3 = "arith.negf"(%1) <{fastmath = #arith.fastmath<none>}> : (f64) -> f64
}) : () -> ()
