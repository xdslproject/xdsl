// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 0 : index} : () -> index
  %1 = "arith.constant"() {"value" = 1000 : index} : () -> index
  %2 = "arith.constant"() {"value" = 3 : index} : () -> index
  %3 = "arith.constant"() {"value" = 10.2 : f32} : () -> f32
  %4 = "arith.constant"() {"value" = 18.1 : f32} : () -> f32
  %5 = "scf.parallel"(%0, %1, %2, %3) ({
    ^0(%8 : index):
      scf.reduce(%4 : f32) {
      ^1(%9 : f32, %10 : f32):
        %11 = "arith.addf"(%9, %10) : (f32, f32) -> f32
        scf.reduce.return %11 : f32
      }
    }) {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : (index, index, index, f32) -> f64
}) : () -> ()

// CHECK: Miss match on scf.parallel result type and reduction op type number 0 , parallel argment is of type f64 whereas reduction operation is of type f32
