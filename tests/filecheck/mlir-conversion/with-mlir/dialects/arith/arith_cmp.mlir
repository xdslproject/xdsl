// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt | filecheck %s

"builtin.module"() ({
  %1 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
  %2 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
  %3 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %4 = "arith.constant"() {"value" = 2 : i32} : () -> i32
  %5 = "arith.cmpf"(%1, %2) {"predicate" = 2 : i64} : (f64, f64) -> i1
  %6 = "arith.select"(%5, %1, %2) : (i1, f64, f64) -> f64
  %7 = "arith.cmpi"(%3, %4) {"predicate" = 1 : i64} : (i32, i32) -> i1
  %8 = "arith.select"(%7, %3, %4) : (i1, i32, i32) -> i32
}) : ()->()

// CHECK:        "arith.cmpf"(%0, %1) {"predicate" = 2 : i64} : (f64, f64) -> i1
// CHECK:        "arith.select"(%4, %0, %1) : (i1, f64, f64) -> f64
// CHECK:        "arith.cmpi"(%2, %3) {"predicate" = 1 : i64} : (i32, i32) -> i1
// CHECK:        "arith.select"(%6, %2, %3) : (i1, i32, i32) -> i32
