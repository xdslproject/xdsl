// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
  %1 = "arith.constant"() {"value" = 1.0 : f64} : () -> f64
  %2 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %3 = "arith.constant"() {"value" = 2 : i32} : () -> i32
  %4 = "arith.constant"() {"value" = 1 : index} : () -> index
  %5 = "arith.constant"() {"value" = 2 : index} : () -> index
  %6 = "arith.cmpf"(%0, %1) {"predicate" = 2 : i64} : (f64, f64) -> i1
  %7 = "arith.select"(%6, %0, %1) : (i1, f64, f64) -> f64
  %8 = "arith.cmpi"(%2, %3) {"predicate" = 1 : i64} : (i32, i32) -> i1
  %9 = "arith.select"(%8, %2, %3) : (i1, i32, i32) -> i32
  %10 = "arith.cmpi"(%4, %5) {"predicate" = 1 : i64} : (index, index) -> i1
  %11 = "arith.select"(%10, %4, %5) : (i1, index, index) -> index
}) : ()->()

// CHECK:        "arith.cmpf"(%0, %1) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f64, f64) -> i1
// CHECK:        "arith.select"(%6, %0, %1) : (i1, f64, f64) -> f64
// CHECK:        "arith.cmpi"(%2, %3) <{predicate = 1 : i64}> : (i32, i32) -> i1
// CHECK:        "arith.select"(%8, %2, %3) : (i1, i32, i32) -> i32
// CHECK:        "arith.cmpi"(%4, %5) <{predicate = 1 : i64}> : (index, index) -> i1
// CHECK:        "arith.select"(%10, %4, %5) : (i1, index, index) -> index
