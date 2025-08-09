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
  cf.br ^bb0(%0 : i32)
^bb0(%1 : i32):
  return %1 : i32
}

/// Test that pass-through successors of BranchOp get folded.

// CHECK:      func.func @br_passthrough(%arg0 : i32, %arg1 : i32) -> (i32, i32) {
// CHECK-NEXT:   "test.termop"() [^bb[[#b0:]], ^bb[[#b1:]], ^bb[[#b2:]]] : () -> ()
// CHECK-NEXT: ^bb[[#b0]]:
// CHECK-NEXT:   cf.br ^bb[[#b2]](%arg0, %arg1 : i32, i32)
// CHECK-NEXT: ^bb[[#b1]](%arg2 : i32):
// CHECK-NEXT:   cf.br ^bb[[#b2]](%arg2, %arg1 : i32, i32)
// CHECK-NEXT: ^bb[[#b2]](%arg4 : i32, %arg5 : i32):
// CHECK-NEXT:   func.return %arg4, %arg5 : i32, i32
// CHECK-NEXT: }
func.func @br_passthrough(%arg0 : i32, %arg1 : i32) -> (i32, i32) {
  "test.termop"() [^bb0, ^bb1, ^bb2] : () -> ()
^bb0:
  cf.br ^bb1(%arg0 : i32)

^bb1(%arg2 : i32):
  cf.br ^bb2(%arg2, %arg1 : i32, i32)

^bb2(%arg4 : i32, %arg5 : i32):
  return %arg4, %arg5 : i32, i32
}

/// Test that dead branches don't affect passthrough
// CHECK:      func.func @br_dead_passthrough() {
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @br_dead_passthrough() {
  cf.br ^bb1
^bb0:
  cf.br ^bb1
^bb1:
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
// CHECK-NEXT:   cf.cond_br %cond, ^bb[[#b0:]], ^bb[[#b1:]]
// CHECK-NEXT: ^bb[[#b0]]:
// CHECK-NEXT:   "test.op"() : () -> ()
// CHECK-NEXT:   cf.br ^bb[[#b1]]
// CHECK-NEXT: ^bb[[#b1]]:
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

/// Test folding conditional branches that are successors of conditional
/// branches with the same condition.
// CHECK:      func.func @cond_br_from_cond_br_with_same_condition(%cond : i1) {
// CHECK-NEXT:   cf.cond_br %cond, ^bb0, ^bb1
// CHECK-NEXT: ^bb0:
// CHECK-NEXT:   func.return
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   "test.termop"() : () -> ()
// CHECK-NEXT: }
func.func @cond_br_from_cond_br_with_same_condition(%cond : i1) {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  cf.cond_br %cond, ^bb3, ^bb2
^bb2:
  "test.termop"() : () -> ()
^bb3:
  return
}

// CHECK:      func.func @branchCondProp(%arg0 : i1) {
// CHECK-NEXT:   %arg0_1 = arith.constant true
// CHECK-NEXT:   %arg0_2 = arith.constant false
// CHECK-NEXT:   cf.cond_br %arg0, ^trueB, ^falseB
// CHECK-NEXT: ^trueB:
// CHECK-NEXT:   "test.op"(%arg0_1) : (i1) -> ()
// CHECK-NEXT:   cf.br ^exit
// CHECK-NEXT: ^falseB:
// CHECK-NEXT:   "test.op"(%arg0_2) : (i1) -> ()
// CHECK-NEXT:   cf.br ^exit
// CHECK-NEXT: ^exit:
// CHECK-NEXT:   func.return
// CHECK-NEXT: }
func.func @branchCondProp(%arg0: i1) {
  cf.cond_br %arg0, ^trueB, ^falseB
^trueB:
  "test.op"(%arg0) : (i1) -> ()
  cf.br ^exit
^falseB:
  "test.op"(%arg0) : (i1) -> ()
  cf.br ^exit
^exit:
  return
}

/// Test the folding of SwitchOp
// CHECK:      func.func @switch_only_default(%flag : i32, %caseOperand0 : f32) {
// CHECK-NEXT:   "test.termop"() [^bb0, ^bb1] : () -> ()
// CHECK-NEXT: ^bb0:
// CHECK-NEXT:   cf.br ^bb1(%caseOperand0 : f32)
// CHECK-NEXT: ^bb1(%arg : f32):
// CHECK-NEXT:   "test.termop"(%arg) : (f32) -> ()
// CHECK-NEXT: }
func.func @switch_only_default(%flag : i32, %caseOperand0 : f32) {
  // add predecessors for all blocks to avoid other canonicalizations.
  "test.termop"() [^bb0, ^bb1] : () -> ()
  ^bb0:
    cf.switch %flag : i32, [
      default: ^bb1(%caseOperand0 : f32)
    ]
  ^bb1(%arg : f32):
    "test.termop"(%arg) : (f32) -> ()
}


// CHECK:      func.func @switch_case_matching_default(%flag : i32, %caseOperand0 : f32, %caseOperand1 : f32) {
// CHECK-NEXT:   "test.termop"() [^bb0, ^bb1, ^bb2] : () -> ()
// CHECK-NEXT: ^bb0:
// CHECK-NEXT:   cf.switch %flag : i32, [
// CHECK-NEXT:     default: ^bb1(%caseOperand0 : f32),
// CHECK-NEXT:     10: ^bb2(%caseOperand1 : f32)
// CHECK-NEXT:   ]
// CHECK-NEXT: ^bb1(%arg : f32):
// CHECK-NEXT:   "test.termop"(%arg) : (f32) -> ()
// CHECK-NEXT: ^bb2(%arg2 : f32):
// CHECK-NEXT:   "test.termop"(%arg2) : (f32) -> ()
// CHECK-NEXT: }
func.func @switch_case_matching_default(%flag : i32, %caseOperand0 : f32, %caseOperand1 : f32) {
  // add predecessors for all blocks to avoid other canonicalizations.
  "test.termop"() [^bb0, ^bb1, ^bb2] : () -> ()
  ^bb0:
    cf.switch %flag : i32, [
      default: ^bb1(%caseOperand0 : f32),
      42: ^bb1(%caseOperand0 : f32),
      10: ^bb2(%caseOperand1 : f32),
      17: ^bb1(%caseOperand0 : f32)
    ]
  ^bb1(%arg : f32):
    "test.termop"(%arg) : (f32) -> ()
  ^bb2(%arg2 : f32):
    "test.termop"(%arg2) : (f32) -> ()
}


// CHECK:      func.func @switch_on_const_no_match(%caseOperand0 : f32, %caseOperand1 : f32, %caseOperand2 : f32) {
// CHECK-NEXT:   "test.termop"() [^bb0, ^bb1, ^bb2, ^bb3] : () -> ()
// CHECK-NEXT: ^bb0:
// CHECK-NEXT:   cf.br ^bb1(%caseOperand0 : f32)
// CHECK-NEXT: ^bb1(%arg : f32):
// CHECK-NEXT:   "test.termop"(%arg) : (f32) -> ()
// CHECK-NEXT: ^bb2(%arg2 : f32):
// CHECK-NEXT:   "test.termop"(%arg2) : (f32) -> ()
// CHECK-NEXT: ^bb3(%arg3 : f32):
// CHECK-NEXT:   "test.termop"(%arg3) : (f32) -> ()
// CHECK-NEXT: }
func.func @switch_on_const_no_match(%caseOperand0 : f32, %caseOperand1 : f32, %caseOperand2 : f32) {
  // add predecessors for all blocks to avoid other canonicalizations.
  "test.termop"() [^bb0, ^bb1, ^bb2, ^bb3] : () -> ()
  ^bb0:
    %c0_i32 = arith.constant 0 : i32
    cf.switch %c0_i32 : i32, [
      default: ^bb1(%caseOperand0 : f32),
      -1: ^bb2(%caseOperand1 : f32),
      1: ^bb3(%caseOperand2 : f32)
    ]
  ^bb1(%arg : f32):
    "test.termop"(%arg) : (f32) -> ()
  ^bb2(%arg2 : f32):
    "test.termop"(%arg2) : (f32) -> ()
  ^bb3(%arg3 : f32):
    "test.termop"(%arg3) : (f32) -> ()
}

// CHECK:      func.func @switch_on_const_with_match(%caseOperand0 : f32, %caseOperand1 : f32, %caseOperand2 : f32) {
// CHECK-NEXT:   "test.termop"() [^bb0, ^bb1, ^bb2, ^bb3] : () -> ()
// CHECK-NEXT: ^bb0:
// CHECK-NEXT:   cf.br ^bb3(%caseOperand2 : f32)
// CHECK-NEXT: ^bb1(%arg : f32):
// CHECK-NEXT:   "test.termop"(%arg) : (f32) -> ()
// CHECK-NEXT: ^bb2(%arg2 : f32):
// CHECK-NEXT:   "test.termop"(%arg2) : (f32) -> ()
// CHECK-NEXT: ^bb3(%arg3 : f32):
// CHECK-NEXT:   "test.termop"(%arg3) : (f32) -> ()
// CHECK-NEXT: }
func.func @switch_on_const_with_match(%caseOperand0 : f32, %caseOperand1 : f32, %caseOperand2 : f32) {
  // add predecessors for all blocks to avoid other canonicalizations.
  "test.termop"() [^bb0, ^bb1, ^bb2, ^bb3] : () -> ()
  ^bb0:
    %c0_i32 = arith.constant 1 : i32
    cf.switch %c0_i32 : i32, [
      default: ^bb1(%caseOperand0 : f32),
      -1: ^bb2(%caseOperand1 : f32),
      1: ^bb3(%caseOperand2 : f32)
    ]
  ^bb1(%arg : f32):
    "test.termop"(%arg) : (f32) -> ()
  ^bb2(%arg2 : f32):
    "test.termop"(%arg2) : (f32) -> ()
  ^bb3(%arg3 : f32):
    "test.termop"(%arg3) : (f32) -> ()
}

// CHECK:      func.func @switch_passthrough(%flag : i32, %caseOperand0 : f32, %caseOperand1 : f32, %caseOperand2 : f32, %caseOperand3 : f32) {
// CHECK-NEXT:   "test.termop"() [^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5] : () -> ()
// CHECK-NEXT: ^bb0:
// CHECK-NEXT:   cf.switch %flag : i32, [
// CHECK-NEXT:     default: ^bb4(%caseOperand0 : f32),
// CHECK-NEXT:     43: ^bb5(%caseOperand1 : f32),
// CHECK-NEXT:     44: ^bb3(%caseOperand2 : f32)
// CHECK-NEXT:   ]
// CHECK-NEXT: ^bb1(%arg : f32):
// CHECK-NEXT:   cf.br ^bb4(%arg : f32)
// CHECK-NEXT: ^bb2(%arg2 : f32):
// CHECK-NEXT:   cf.br ^bb5(%arg2 : f32)
// CHECK-NEXT: ^bb3(%arg3 : f32):
// CHECK-NEXT:   "test.termop"(%arg3) : (f32) -> ()
// CHECK-NEXT: ^bb4(%arg4 : f32):
// CHECK-NEXT:   "test.termop"(%arg4) : (f32) -> ()
// CHECK-NEXT: ^bb5(%arg5 : f32):
// CHECK-NEXT:   "test.termop"(%arg5) : (f32) -> ()
// CHECK-NEXT: }
func.func @switch_passthrough(%flag : i32,
                         %caseOperand0 : f32,
                         %caseOperand1 : f32,
                         %caseOperand2 : f32,
                         %caseOperand3 : f32) {
  // add predecessors for all blocks to avoid other canonicalizations.
  "test.termop"() [^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5] : () -> ()
  ^bb0:
    cf.switch %flag : i32, [
      default: ^bb1(%caseOperand0 : f32),
      43: ^bb2(%caseOperand1 : f32),
      44: ^bb3(%caseOperand2 : f32)
    ]
  ^bb1(%arg : f32):
    cf.br ^bb4(%arg : f32)
  ^bb2(%arg2 : f32):
    cf.br ^bb5(%arg2 : f32)
  ^bb3(%arg3 : f32):
    "test.termop"(%arg3) : (f32) -> ()
  ^bb4(%arg4 : f32):
    "test.termop"(%arg4) : (f32) -> ()
  ^bb5(%arg5 : f32):
    "test.termop"(%arg5) : (f32) -> ()
}

// CHECK:      func.func @switch_from_switch_with_same_value_with_match(%flag : i32, %caseOperand0 : f32, %caseOperand1 : f32) {
// CHECK-NEXT:   "test.termop"() [^bb0, ^bb1, ^bb2, ^bb3] : () -> ()
// CHECK-NEXT: ^bb0:
// CHECK-NEXT:   cf.switch %flag : i32, [
// CHECK-NEXT:     default: ^bb1,
// CHECK-NEXT:     42: ^bb4
// CHECK-NEXT:   ]
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   "test.termop"() : () -> ()
// CHECK-NEXT: ^bb4:
// CHECK-NEXT:   "test.op"() : () -> ()
// CHECK-NEXT:   cf.br ^bb3(%caseOperand1 : f32)
// CHECK-NEXT: ^bb2(%arg3 : f32):
// CHECK-NEXT:   "test.termop"(%arg3) : (f32) -> ()
// CHECK-NEXT: ^bb3(%arg4 : f32):
// CHECK-NEXT:   "test.termop"(%arg4) : (f32) -> ()
// CHECK-NEXT: }
func.func @switch_from_switch_with_same_value_with_match(%flag : i32, %caseOperand0 : f32, %caseOperand1 : f32) {
  // add predecessors for all blocks except ^bb2 to avoid other canonicalizations.
  "test.termop"() [^bb0, ^bb1, ^bb3, ^bb4] : () -> ()
  ^bb0:
    cf.switch %flag : i32, [
      default: ^bb1,
      42: ^bb2
    ]

  ^bb1:
    "test.termop"() : () -> ()
  ^bb2:
    // prevent this block from being simplified away
    "test.op"() : () -> ()
    cf.switch %flag : i32, [
      default: ^bb3(%caseOperand0 : f32),
      42: ^bb4(%caseOperand1 : f32)
    ]
  ^bb3(%arg3 : f32):
    "test.termop"(%arg3) : (f32) -> ()
  ^bb4(%arg4 : f32):
    "test.termop"(%arg4) : (f32) -> ()
}

// CHECK:      func.func @switch_from_switch_with_same_value_no_match(%flag : i32, %caseOperand0 : f32, %caseOperand1 : f32, %caseOperand2 : f32) {
// CHECK-NEXT:   "test.termop"() [^bb0, ^bb1, ^bb2, ^bb3, ^bb4] : () -> ()
// CHECK-NEXT: ^bb0:
// CHECK-NEXT:   cf.switch %flag : i32, [
// CHECK-NEXT:     default: ^bb1,
// CHECK-NEXT:     42: ^bb5
// CHECK-NEXT:   ]
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   "test.termop"() : () -> ()
// CHECK-NEXT: ^bb5:
// CHECK-NEXT:   "test.op"() : () -> ()
// CHECK-NEXT:   cf.br ^bb2(%caseOperand0 : f32)
// CHECK-NEXT: ^bb2(%arg3 : f32):
// CHECK-NEXT:   "test.termop"(%arg3) : (f32) -> ()
// CHECK-NEXT: ^bb3(%arg4 : f32):
// CHECK-NEXT:   "test.termop"(%arg4) : (f32) -> ()
// CHECK-NEXT: ^bb4(%arg5 : f32):
// CHECK-NEXT:   "test.termop"(%arg5) : (f32) -> ()
// CHECK-NEXT: }
func.func @switch_from_switch_with_same_value_no_match(%flag : i32, %caseOperand0 : f32, %caseOperand1 : f32, %caseOperand2 : f32) {
  // add predecessors for all blocks except ^bb2 to avoid other canonicalizations.
  "test.termop"() [^bb0, ^bb1, ^bb3, ^bb4, ^bb5] : () -> ()
  ^bb0:
    cf.switch %flag : i32, [
      default: ^bb1,
      42: ^bb2
    ]
  ^bb1:
    "test.termop"() : () -> ()
  ^bb2:
    "test.op"() : () -> ()
    cf.switch %flag : i32, [
      default: ^bb3(%caseOperand0 : f32),
      0: ^bb4(%caseOperand1 : f32),
      43: ^bb5(%caseOperand2 : f32)
    ]
  ^bb3(%arg3 : f32):
    "test.termop"(%arg3) : (f32) -> ()
  ^bb4(%arg4 : f32):
    "test.termop"(%arg4) : (f32) -> ()
  ^bb5(%arg5 : f32):
    "test.termop"(%arg5) : (f32) -> ()
}

// CHECK:      func.func @switch_from_switch_default_with_same_value(%flag : i32, %caseOperand0 : f32, %caseOperand1 : f32, %caseOperand2 : f32) {
// CHECK-NEXT:   "test.termop"() [^bb0, ^bb1, ^bb2, ^bb3, ^bb4] : () -> ()
// CHECK-NEXT: ^bb0:
// CHECK-NEXT:   cf.switch %flag : i32, [
// CHECK-NEXT:     default: ^bb5,
// CHECK-NEXT:     42: ^bb1
// CHECK-NEXT:   ]
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   "test.termop"() : () -> ()
// CHECK-NEXT: ^bb5:
// CHECK-NEXT:   "test.op"() : () -> ()
// CHECK-NEXT:   cf.switch %flag : i32, [
// CHECK-NEXT:     default: ^bb2(%caseOperand0 : f32),
// CHECK-NEXT:     43: ^bb4(%caseOperand2 : f32)
// CHECK-NEXT:   ]
// CHECK-NEXT: ^bb2(%arg3 : f32):
// CHECK-NEXT:   "test.termop"(%arg3) : (f32) -> ()
// CHECK-NEXT: ^bb3(%arg4 : f32):
// CHECK-NEXT:   "test.termop"(%arg4) : (f32) -> ()
// CHECK-NEXT: ^bb4(%arg5 : f32):
// CHECK-NEXT:   "test.termop"(%arg5) : (f32) -> ()
// CHECK-NEXT: }
func.func @switch_from_switch_default_with_same_value(%flag : i32, %caseOperand0 : f32, %caseOperand1 : f32, %caseOperand2 : f32) {
  // add predecessors for all blocks except ^bb2 to avoid other canonicalizations.
  "test.termop"() [^bb0, ^bb1, ^bb3, ^bb4, ^bb5] : () -> ()
  ^bb0:
    cf.switch %flag : i32, [
      default: ^bb2,
      42: ^bb1
    ]
  ^bb1:
    "test.termop"() : () -> ()
  ^bb2:
    "test.op"() : () -> ()
    cf.switch %flag : i32, [
      default: ^bb3(%caseOperand0 : f32),
      42: ^bb4(%caseOperand1 : f32),
      43: ^bb5(%caseOperand2 : f32)
    ]
  ^bb3(%arg3 : f32):
    "test.termop"(%arg3) : (f32) -> ()
  ^bb4(%arg4 : f32):
    "test.termop"(%arg4) : (f32) -> ()
  ^bb5(%arg5 : f32):
    "test.termop"(%arg5) : (f32) -> ()
}
