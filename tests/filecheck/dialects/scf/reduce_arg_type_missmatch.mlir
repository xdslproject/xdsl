// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 0 : index} : () -> index
  %1 = "arith.constant"() {"value" = 1000 : index} : () -> index
  %2 = "arith.constant"() {"value" = 3 : index} : () -> index
  %3 = "arith.constant"() {"value" = 10.2 : f32} : () -> f32
  %4 = "arith.constant"() {"value" = 18.1 : f32} : () -> f32
  %5 = "scf.parallel"(%0, %1, %2, %3) ({
    ^0(%8 : index):
      "scf.reduce"(%4) ({
      ^1(%9 : f64, %10 : f64):
        %11 = "arith.addf"(%9, %10) : (f64, f64) -> f64
        "scf.reduce.return"(%11) : (f64) -> ()
      }) : (f32) -> ()
    }) {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : (index, index, index, f32) -> f32
}) : () -> ()

// CHECK: scf.reduce block argument types must match the operand type  but have f64 and f32
