// RUN: xdsl-opt %s | mlir-opt --allow-unregistered-dialect | filecheck %s

"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {"value" = 0 : i64} : () -> i64
    %1 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> f32
    %2 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> i32
    %3 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> i64
    %4 = "builtin.unrealized_conversion_cast"() {"comment" = "test"} : () -> i64
    %5 = "builtin.unrealized_conversion_cast"(%0, %0) : (i64, i64) -> f32
    %6, %7 = "builtin.unrealized_conversion_cast"(%5) : (f32) -> (i64, i64)
    "func.return"() : () -> ()
  }) {"function_type" = () -> (), "sym_name" = "builtin"} : () -> ()
}) : () -> ()

// CHECK: module {
// CHECK-NEXT:   func.func @builtin() {
// CHECK-NEXT:     %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:     %0 = builtin.unrealized_conversion_cast %c0_i64 : i64 to f32
// CHECK-NEXT:     %1 = builtin.unrealized_conversion_cast %c0_i64 : i64 to i32
// CHECK-NEXT:     %2 = builtin.unrealized_conversion_cast %c0_i64 : i64 to i64
// CHECK-NEXT:     %3 = builtin.unrealized_conversion_cast to i64 {comment = "test"}
// CHECK-NEXT:     %4 = builtin.unrealized_conversion_cast %c0_i64, %c0_i64 : i64, i64 to f32
// CHECK-NEXT:     %5:2 = builtin.unrealized_conversion_cast %4 : f32 to i64, i64
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }
