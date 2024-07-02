// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%q0, %q1 = qssa.alloc<2>

// CHECK: %q0, %q1 = qssa.alloc<2>

%q2 = qssa.h %q0

// CHECK-NEXT: %q2 = qssa.h %q0

%q3 = qssa.rz <pi/2> %q1

// CHECK-NEXT: %q3 = qssa.rz <pi/2> %q1

%q4, %q5 = qssa.cnot %q2, %q3

// CHECK-NEXT: %q4, %q5 = qssa.cnot %q2, %q3

%0 = qssa.measure %q4

// CHECK-NEXT: %0 = qssa.measure %q4

// CHECK-GENERIC: %q0, %q1 = "qssa.alloc"() : () -> (!qssa.qubit, !qssa.qubit)
// CHECK-GENERIC-NEXT: %q2 = "qssa.h"(%q0) : (!qssa.qubit) -> !qssa.qubit
// CHECK-GENERIC-NEXT: %q3 = "qssa.rz"(%q1) <{"angle" = !quantum.angle<pi/2>}> : (!qssa.qubit) -> !qssa.qubit
// CHECK-GENERIC-NEXT: %q4, %q5 = "qssa.cnot"(%q2, %q3) : (!qssa.qubit, !qssa.qubit) -> (!qssa.qubit, !qssa.qubit)
// CHECK-GENERIC-NEXT: %0 = "qssa.measure"(%q4) : (!qssa.qubit) -> i1
