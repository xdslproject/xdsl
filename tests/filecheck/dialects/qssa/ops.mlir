// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%q0 = "test.op"() : () -> !qssa.qubits<1>

// CHECK: %q0 = "test.op"() : () -> !qssa.qubits<1>

%0 = qssa.h %q0

// CHECK-NEXT: %0 = qssa.h %q0

// CHECK-GENERIC: %q0 = "test.op"() : () -> !qssa.qubits<1>
// CHECK-GENERIC-NEXT: %0 = "qssa.h"(%q0) : (!qssa.qubits<1>) -> !qssa.qubits<1>

