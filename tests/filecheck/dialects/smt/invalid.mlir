// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

%arg1 = smt.constant true

%and = smt.or %arg1
// CHECK: expected integer >= 2, got 1

// -----

%arg1 = smt.constant true

%and = smt.and %arg1
// CHECK: expected integer >= 2, got 1

// -----

%arg1 = smt.constant true

%and = smt.xor %arg1
// CHECK: expected integer >= 2, got 1

// -----

%arg1 = smt.constant true

%and = smt.distinct %arg1 : !smt.bool
// CHECK: operand 'inputs' at position 0 does not verify:
// CHECK-NEXT: incorrect length for range variable:
// CHECK-NEXT: expected integer >= 2, got 1

// -----

%arg1 = smt.constant true

%and = smt.eq %arg1 : !smt.bool
// CHECK: operand 'inputs' at position 0 does not verify:
// CHECK-NEXT: incorrect length for range variable:
// CHECK-NEXT: expected integer >= 2, got 1

// -----

%forall = smt.forall {^bb0:}
// CHECK: Operation smt.forall contains empty block in single-block
// CHECK-SAME: region that expects at least a terminator

// -----

%arg1 = smt.constant true
%arg2 = smt.constant false
%forall = smt.forall {
    smt.yield %arg1, %arg2 : !smt.bool, !smt.bool
}
// CHECK: region yield terminator must have a single boolean operand,
// CHECK-SAME: got ('!smt.bool', '!smt.bool')

// -----

%arg1 = smt.constant true

%exists = smt.exists {^bb0:}
// CHECK: Operation smt.exists contains empty block in single-block
// CHECK-SAME: region that expects at least a terminator

// -----

%arg1 = smt.constant true
%arg2 = smt.constant false
%exists = smt.exists {
    smt.yield %arg1, %arg2 : !smt.bool, !smt.bool
}
// CHECK: region yield terminator must have a single boolean operand,
// CHECK-SAME: got ('!smt.bool', '!smt.bool')
