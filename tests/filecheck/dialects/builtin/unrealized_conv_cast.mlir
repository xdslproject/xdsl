// RUN: XDSL_ROUNDTRIP

// CHECK: module
"builtin.module"() ({
  "func.func"() ({
    // CHECK: %0 = arith.constant 0 : i64
    %0 = "arith.constant"() {"value" = 0 : i64} : () -> i64
    // CHECK: %1 = builtin.unrealized_conversion_cast %0 : i64 to f32
    %1 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> f32
    // CHECK: %2 = builtin.unrealized_conversion_cast %0 : i64 to i32
    %2 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> i32
    // CHECK: %3 = builtin.unrealized_conversion_cast %0 : i64 to i64
    %3 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> i64
    // CHECK: %4 = builtin.unrealized_conversion_cast to i64  {comment = "test"}
    %4 = "builtin.unrealized_conversion_cast"() {"comment" = "test"} : () -> i64
    // CHECK: %5 = builtin.unrealized_conversion_cast %0, %0 : i64, i64 to f32
    %5 = "builtin.unrealized_conversion_cast"(%0, %0) : (i64, i64) -> f32
    // CHECK: %6, %7 = builtin.unrealized_conversion_cast %5 : f32 to i64, i64
    %6, %7 = "builtin.unrealized_conversion_cast"(%5) : (f32) -> (i64, i64)
    // CHECK: %8 = builtin.unrealized_conversion_cast to none  {comment = "test"}
    %8 = "builtin.unrealized_conversion_cast"() {"comment" = "test"} : () -> none
    "func.return"() : () -> ()
  }) {"function_type" = () -> (), "sym_name" = "builtin"} : () -> ()
}) : () -> ()
