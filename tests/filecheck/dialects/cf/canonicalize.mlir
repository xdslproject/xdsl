// RUN: xdsl-opt -p canonicalize %s | filecheck %s

// CHECK:      func.func @assert_true() -> i1 {
// CHECK-NEXT:   %0 = arith.constant true
// CHECK-NEXT:   func.return %0 : i1
// CHECK-NEXT: }

func.func @assert_true() -> i1 {
  %0 = arith.constant true
  cf.assert %0 , "assert true"
  func.return %0 : i1
}

/// Test the folding of BranchOp.

// CHECK:      func.func @br_folding() -> i32 {
// CHECK-NEXT:   %[[#v0:]] = arith.constant 0 : i32
// CHECK-NEXT:   func.return %[[#v0]] : i32
// CHECK-NEXT: }
func.func @br_folding() -> i32 {
  %0 = arith.constant 0 : i32
  cf.br ^0(%0 : i32)
^0(%1 : i32):
  return %1 : i32
}

/// Test that pass-through successors of BranchOp get folded.

// CHECK:      func.func @br_passthrough(%arg0 : i32, %arg1 : i32) -> (i32, i32) {
// CHECK-NEXT:   "test.termop"() [^[[#b0:]], ^[[#b1:]], ^[[#b2:]]] : () -> ()
// CHECK-NEXT: ^[[#b0]]:
// CHECK-NEXT:   cf.br ^[[#b2]](%arg0, %arg1 : i32, i32)
// CHECK-NEXT: ^[[#b1]](%arg2 : i32):
// CHECK-NEXT:   cf.br ^[[#b2]](%arg2, %arg1 : i32, i32)
// CHECK-NEXT: ^[[#b2]](%arg4 : i32, %arg5 : i32):
// CHECK-NEXT:   func.return %arg4, %arg5 : i32, i32
// CHECK-NEXT: }
func.func @br_passthrough(%arg0 : i32, %arg1 : i32) -> (i32, i32) {
  "test.termop"() [^0, ^1, ^2] : () -> ()
^0:
  cf.br ^1(%arg0 : i32)

^1(%arg2 : i32):
  cf.br ^2(%arg2, %arg1 : i32, i32)

^2(%arg4 : i32, %arg5 : i32):
  return %arg4, %arg5 : i32, i32
}

/// Test that dead branches don't affect passthrough
// CHECK:      func.func @br_dead_passthrough() {
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @br_dead_passthrough() {
cf.br ^1
^0:
cf.br ^1
^1:
func.return
}
