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

/// Test the folding of CondBranchOp with a constant condition.
/// This will reduce further with other rewrites

// CHECK:      func.func @cond_br_folding(%cond : i1, %a : i32) {
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @cond_br_folding(%cond : i1, %a : i32) {
  %false_cond = arith.constant false
  %true_cond = arith.constant true
  cf.cond_br %cond, ^bb1, ^bb2(%a : i32)

^bb1:
  cf.cond_br %true_cond, ^bb3, ^bb2(%a : i32)

^bb2(%x : i32):
  cf.cond_br %false_cond, ^bb2(%x : i32), ^bb3

^bb3:
  return
}

/// Test the compound folding of BranchOp and CondBranchOp.
// CHECK-NEXT: func.func @cond_br_and_br_folding(%a : i32) {
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @cond_br_and_br_folding(%a : i32) {

  %false_cond = arith.constant false
  %true_cond = arith.constant true
  cf.cond_br %true_cond, ^bb2, ^bb1(%a : i32)

^bb1(%x : i32):
  cf.cond_br %false_cond, ^bb1(%x : i32), ^bb2

^bb2:
  return
}

/// Test that pass-through successors of CondBranchOp get folded.
// CHECK:      func.func @cond_br_passthrough(%arg0 : i32, %arg1 : i32, %arg2 : i32, %cond : i1) -> (i32, i32) {
// CHECK-NEXT:   %arg4 = arith.select %cond, %arg0, %arg2 : i32
// CHECK-NEXT:   %arg5 = arith.select %cond, %arg1, %arg2 : i32
// CHECK-NEXT:   func.return %arg4, %arg5 : i32, i32
// CHECK-NEXT: }
func.func @cond_br_passthrough(%arg0 : i32, %arg1 : i32, %arg2 : i32, %cond : i1) -> (i32, i32) {
  cf.cond_br %cond, ^bb1(%arg0 : i32), ^bb2(%arg2, %arg2 : i32, i32)
^bb1(%arg3: i32):
  cf.br ^bb2(%arg3, %arg1 : i32, i32)
^bb2(%arg4: i32, %arg5: i32):
  return %arg4, %arg5 : i32, i32
}

/// Test the failure modes of collapsing CondBranchOp pass-throughs successors.

// CHECK-NEXT: func.func @cond_br_pass_through_fail(%cond : i1) {
// CHECK-NEXT:   cf.cond_br %cond, ^[[#b0:]], ^[[#b1:]]
// CHECK-NEXT: ^[[#b0]]:
// CHECK-NEXT:   "test.op"() : () -> ()
// CHECK-NEXT:   cf.br ^[[#b1]]
// CHECK-NEXT: ^[[#b1]]:
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @cond_br_pass_through_fail(%cond : i1) {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  "test.op"() : () -> ()
  cf.br ^bb2
^bb2:
  return
}

/// Test the folding of CondBranchOp when the successors are identical.
// CHECK:      func.func @cond_br_same_successor(%cond : i1, %a : i32) {
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @cond_br_same_successor(%cond : i1, %a : i32) {
  cf.cond_br %cond, ^bb1(%a : i32), ^bb1(%a : i32)
^bb1(%result : i32):
  return
}

/// Test the folding of CondBranchOp when the successors are identical, but the
/// arguments are different.
// CHECK:      func.func @cond_br_same_successor_insert_select(%cond : i1, %a : i32, %b : i32, %c : tensor<2xi32>, %d : tensor<2xi32>) -> (i32, tensor<2xi32>) {
// CHECK-NEXT:   %result = arith.select %cond, %a, %b : i32
// CHECK-NEXT:   %result2 = arith.select %cond, %c, %d : tensor<2xi32>
// CHECK-NEXT:   func.return %result, %result2 : i32, tensor<2xi32>
// CHECK-NEXT: }
func.func @cond_br_same_successor_insert_select(
      %cond : i1, %a : i32, %b : i32, %c : tensor<2xi32>, %d : tensor<2xi32>
    ) -> (i32, tensor<2xi32>)  {
  cf.cond_br %cond, ^bb1(%a, %c : i32, tensor<2xi32>), ^bb1(%b, %d : i32, tensor<2xi32>)
^bb1(%result : i32, %result2 : tensor<2xi32>):
  return %result, %result2 : i32, tensor<2xi32>
}
