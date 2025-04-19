// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

%arg1 = smt.constant true

%and = smt.or %arg1
// CHECK: operand at position 0 does not verify:
// CHECK-NEXT: incorrect length for range variable:
// CHECK-NEXT: expected integer >= 2, got 1

// -----

%arg1 = smt.constant true

%and = smt.and %arg1
// CHECK: operand at position 0 does not verify:
// CHECK-NEXT: incorrect length for range variable:
// CHECK-NEXT: expected integer >= 2, got 1

// -----

%arg1 = smt.constant true

%and = smt.xor %arg1
// CHECK: operand at position 0 does not verify:
// CHECK-NEXT: incorrect length for range variable:
// CHECK-NEXT: expected integer >= 2, got 1

// -----

%arg1 = smt.constant true

%and = smt.distinct %arg1 : !smt.bool
// CHECK: operand at position 0 does not verify:
// CHECK-NEXT: incorrect length for range variable:
// CHECK-NEXT: expected integer >= 2, got 1

// -----

%arg1 = smt.constant true

%and = smt.eq %arg1 : !smt.bool
// CHECK: operand at position 0 does not verify:
// CHECK-NEXT: incorrect length for range variable:
// CHECK-NEXT: expected integer >= 2, got 1
