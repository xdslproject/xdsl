"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "builtin"}> ({
    %0 = "arith.constant"() <{value = 0 : i64}> : () -> i64
    %1 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> f32
    %2 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> i32
    %3 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> i64
    %4 = "builtin.unrealized_conversion_cast"() {comment = "test"} : () -> i64
    %5 = "builtin.unrealized_conversion_cast"(%0, %0) : (i64, i64) -> f32
    %6:2 = "builtin.unrealized_conversion_cast"(%5) : (f32) -> (i64, i64)
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
