// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 0 : i64} : () -> i64
  %1 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> i32
  %2 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> f32
  %3 = "arith.constant"() {"value" = 10.4 : f32} : () -> f32
  %4 = "builtin.unrealized_conversion_cast"(%3) : (f32) -> f64
  %5 = "builtin.unrealized_conversion_cast"(%3) : (f32) -> i64
  %6 = "builtin.unrealized_conversion_cast"(%3) : (f32) -> f32
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %0 = "arith.constant"() <{value = 0 : i64}> : () -> i64
// CHECK-NEXT:   %1 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> i32
// CHECK-NEXT:   %2 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> f32
// CHECK-NEXT:   %3 = "arith.constant"() <{value = 1.040000e+01 : f32}> : () -> f32
// CHECK-NEXT:   %4 = "builtin.unrealized_conversion_cast"(%3) : (f32) -> f64
// CHECK-NEXT:   %5 = "builtin.unrealized_conversion_cast"(%3) : (f32) -> i64
// CHECK-NEXT:   %6 = "builtin.unrealized_conversion_cast"(%3) : (f32) -> f32
// CHECK-NEXT: }) : () -> ()
