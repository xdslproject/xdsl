// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%q0, %q1 = qref.alloc<2>

// CHECK: %q0, %q1 = qref.alloc<2>

qref.h %q0

// CHECK-NEXT: qref.h %q0

qref.rz <pi:2> %q1

// CHECK-NEXT: qref.rz <pi:2> %q1

qref.cnot %q0, %q1

// CHECK-NEXT: qref.cnot %q0, %q1

%0 = qref.measure %q0

// CHECK-NEXT: %0 = qref.measure %q0

// CHECK-GENERIC: %q0, %q1 = "qref.alloc"() : () -> (!qref.qubit, !qref.qubit)
// CHECK-GENERIC-NEXT: "qref.h"(%q0) : (!qref.qubit) -> ()
// CHECK-GENERIC-NEXT: "qref.rz"(%q1) <{"angle" = !quantum.angle<pi:2>}> : (!qref.qubit) -> ()
// CHECK-GENERIC-NEXT: "qref.cnot"(%q0, %q1) : (!qref.qubit, !qref.qubit) -> ()
// CHECK-GENERIC-NEXT: %0 = "qref.measure"(%q0) : (!qref.qubit) -> i1
