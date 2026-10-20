// RUN: XDSL_ROUNDTRIP

builtin.module {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1000 : index
  %2 = arith.constant 3 : index
  %3 = arith.constant 1.020000e+01 : f32
  %4 = arith.constant 1.810000e+01 : f32
  %5 = "scf.parallel"(%0, %1, %2, %3) ({
  ^bb0(%6: index):
    scf.reduce(%4 : f32) {
    ^bb1(%7: f32, %8: f32):
      %9 = arith.addf %7, %8 : f32
      scf.reduce.return %9 : f32
    }
  }) {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : (index, index, index, f32) -> f32

  %initial_i32 = arith.constant 2 : i32
  %sum, %product = "scf.parallel"(%0, %1, %2, %3, %initial_i32) ({
  ^bb0(%j: index):
    scf.reduce(%4, %initial_i32 : f32, i32) {
    ^bb0(%flhs: f32, %frhs: f32):
      %fadd = arith.addf %flhs, %frhs : f32
      scf.reduce.return %fadd : f32
    }, {
    ^bb0(%ilhs: i32, %irhs: i32):
      %imul = arith.muli %ilhs, %irhs : i32
      scf.reduce.return %imul : i32
    } {tag = "sum_product"}
  }) {operandSegmentSizes = array<i32: 1, 1, 1, 2>} : (index, index, index, f32, i32) -> (f32, i32)
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 1000 : index
// CHECK-NEXT:   %2 = arith.constant 3 : index
// CHECK-NEXT:   %3 = arith.constant 1.020000e+01 : f32
// CHECK-NEXT:   %4 = arith.constant 1.810000e+01 : f32
// CHECK-NEXT:   %5 = "scf.parallel"(%0, %1, %2, %3) <{operandSegmentSizes = array<i32: 1, 1, 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%6: index):
// CHECK-NEXT:     scf.reduce(%4 : f32) {
// CHECK-NEXT:     ^bb0(%7: f32, %8: f32):
// CHECK-NEXT:       %9 = arith.addf %7, %8 : f32
// CHECK-NEXT:       scf.reduce.return %9 : f32
// CHECK-NEXT:     }
// CHECK-NEXT:   }) : (index, index, index, f32) -> f32
// CHECK-NEXT:   %initial_i32 = arith.constant 2 : i32
// CHECK-NEXT:   %sum, %product = "scf.parallel"(%0, %1, %2, %3, %initial_i32) <{operandSegmentSizes = array<i32: 1, 1, 1, 2>}> ({
// CHECK-NEXT:   ^bb0(%j: index):
// CHECK-NEXT:     scf.reduce(%4, %initial_i32 : f32, i32) {
// CHECK-NEXT:     ^bb0(%flhs: f32, %frhs: f32):
// CHECK-NEXT:       %fadd = arith.addf %flhs, %frhs : f32
// CHECK-NEXT:       scf.reduce.return %fadd : f32
// CHECK-NEXT:     }, {
// CHECK-NEXT:     ^bb0(%ilhs: i32, %irhs: i32):
// CHECK-NEXT:       %imul = arith.muli %ilhs, %irhs : i32
// CHECK-NEXT:       scf.reduce.return %imul : i32
// CHECK-NEXT:     } {tag = "sum_product"}
// CHECK-NEXT:   }) : (index, index, index, f32, i32) -> (f32, i32)
// CHECK-NEXT: }
