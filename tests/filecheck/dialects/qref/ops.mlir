// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%q0, %q1 = qref.alloc<2>

// CHECK: %q0, %q1 = qref.alloc<2>

qref.h %q0

// CHECK-NEXT: qref.h %q0

qref.cz %q1, %q0

// CHECK-NEXT: qref.cz %q1, %q0

qref.cnot %q1, %q0

// CHECK-NEXT: qref.cnot %q1, %q0

%0 = qref.measure %q0

// CHECK-NEXT: %0 = qref.measure %q0

// CHECK-GENERIC: %q0, %q1 = "qref.alloc"() : () -> (!qref.qubit, !qref.qubit)
// CHECK-GENERIC-NEXT: "qref.h"(%q0) : (!qref.qubit) -> ()
// CHECK-GENERIC-NEXT: "qref.cz"(%q1, %q0) : (!qref.qubit, !qref.qubit) -> ()
// CHECK-GENERIC-NEXT: "qref.cnot"(%q1, %q0) : (!qref.qubit, !qref.qubit) -> ()
// CHECK-GENERIC-NEXT: %0 = "qref.measure"(%q0) : (!qref.qubit) -> i1
