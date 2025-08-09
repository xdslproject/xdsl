// RUN: XDSL_ROUNDTRIP

builtin.module {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1000 : index
  %2 = arith.constant 3 : index
  %3 = arith.constant 1.020000e+01 : f32
  %4 = arith.constant 1.810000e+01 : f32
  %5 = "scf.parallel"(%0, %1, %2, %3) ({
  ^bb0(%6 : index):
    scf.reduce(%4 : f32) {
    ^bb1(%7 : f32, %8 : f32):
      %9 = arith.addf %7, %8 : f32
      scf.reduce.return %9 : f32
    }
  }) {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : (index, index, index, f32) -> f32
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 1000 : index
// CHECK-NEXT:   %2 = arith.constant 3 : index
// CHECK-NEXT:   %3 = arith.constant 1.020000e+01 : f32
// CHECK-NEXT:   %4 = arith.constant 1.810000e+01 : f32
// CHECK-NEXT:   %5 = "scf.parallel"(%0, %1, %2, %3) <{operandSegmentSizes = array<i32: 1, 1, 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%6 : index):
// CHECK-NEXT:     scf.reduce(%4 : f32) {
// CHECK-NEXT:     ^bb1(%7 : f32, %8 : f32):
// CHECK-NEXT:       %9 = arith.addf %7, %8 : f32
// CHECK-NEXT:       scf.reduce.return %9 : f32
// CHECK-NEXT:     }
// CHECK-NEXT:   }) : (index, index, index, f32) -> f32
