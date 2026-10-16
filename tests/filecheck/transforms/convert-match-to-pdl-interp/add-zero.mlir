// RUN: cat %p/../../dialects/match/add-zero.mlir %s | xdsl-opt -p convert-match-to-pdl-interp,apply-pdl-interp | filecheck %s

// Reuse the handwritten Match pattern and its PDLInterp rewriter.
// Matching must forward the lhs, not the zero rhs or location operands.
// Interleave successes and failures to exercise repeated matcher invocations.

// CHECK: func.func @rewrite(%x: i32) -> (i32, i32, i32, i32, i32) {
// CHECK-NEXT: %zero = arith.constant 0 : i32
// CHECK-NEXT: %seven = arith.constant 7 : i32
// CHECK-NEXT: %nonzero = arith.addi %x, %seven : i32
// CHECK-NEXT: %block_arg = arith.addi %zero, %x : i32
// CHECK-NEXT: %wrong_name = arith.subi %x, %zero : i32
// CHECK-NEXT: func.return %x, %nonzero, %block_arg, %wrong_name, %nonzero : i32, i32, i32, i32, i32
// CHECK-NEXT: }
func.func @rewrite(%x: i32) -> (i32, i32, i32, i32, i32) {
  %zero = arith.constant 0 : i32
  %seven = arith.constant 7 : i32
  %first = arith.addi %x, %zero : i32
  %nonzero = arith.addi %x, %seven : i32
  %chain = arith.addi %first, %zero : i32
  %block_arg = arith.addi %zero, %x : i32
  %wrong_name = arith.subi %x, %zero : i32
  %last = arith.addi %nonzero, %zero : i32
  func.return %chain, %nonzero, %block_arg, %wrong_name, %last : i32, i32, i32, i32, i32
}
