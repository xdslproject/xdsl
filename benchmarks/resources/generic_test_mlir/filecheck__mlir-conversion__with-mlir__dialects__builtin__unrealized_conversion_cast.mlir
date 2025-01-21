"builtin.module"() ({
  %0 = "arith.constant"() <{value = 0 : i64}> : () -> i64
  %1 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> i32
  %2 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> f32
  %3 = "arith.constant"() <{value = 1.040000e+01 : f32}> : () -> f32
  %4 = "builtin.unrealized_conversion_cast"(%3) : (f32) -> f64
  %5 = "builtin.unrealized_conversion_cast"(%3) : (f32) -> i64
  %6 = "builtin.unrealized_conversion_cast"(%3) : (f32) -> f32
}) : () -> ()
