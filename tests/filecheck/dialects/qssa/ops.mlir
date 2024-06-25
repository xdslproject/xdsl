// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%q0, %q1 = qssa.alloc<2>

// CHECK: %q0, %q1 = qssa.alloc<2>

%q2 = qssa.h %q0

// CHECK-NEXT: %q2 = qssa.h %q0

%q3, %q4 = qssa.cz %q1, %q2

// CHECK-NEXT: %q3, %q4 = qssa.cz %q1, %q2

%q5, %q6 = qssa.cnot %q3, %q4

// CHECK-NEXT: %q5, %q6 = qssa.cnot %q3, %q4

%0 = qssa.measure %q6

// CHECK-NEXT: %0 = qssa.measure %q6

// CHECK-GENERIC: %q0, %q1 = "qssa.alloc"() : () -> (!qssa.qubit, !qssa.qubit)
// CHECK-GENERIC-NEXT: %q2 = "qssa.h"(%q0) : (!qssa.qubit) -> !qssa.qubit
// CHECK-GENERIC-NEXT: %q3, %q4 = "qssa.cz"(%q1, %q2) : (!qssa.qubit, !qssa.qubit) -> (!qssa.qubit, !qssa.qubit)
// CHECK-GENERIC-NEXT: %q5, %q6 = "qssa.cnot"(%q3, %q4) : (!qssa.qubit, !qssa.qubit) -> (!qssa.qubit, !qssa.qubit)
// CHECK-GENERIC-NEXT: %0 = "qssa.measure"(%q6) : (!qssa.qubit) -> i1
