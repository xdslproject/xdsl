// RUN: xdsl-opt %s | xdsl-opt | filecheck %s

// CHECK: module
"builtin.module"() ({
  "func.func"() ({
    // CHECK: %0 = "arith.constant"() {"value" = 0 : i64} : () -> i64
    %0 = "arith.constant"() {"value" = 0 : i64} : () -> i64
    // CHECK: %1 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> f32
    %1 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> f32
    // CHECK: %2 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> i32
    %2 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> i32
    // CHECK: %3 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> i64
    %3 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> i64
    "func.return"() : () -> ()
  }) {"function_type" = () -> (), "sym_name" = "builtin"} : () -> ()
}) : () -> ()
