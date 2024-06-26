// RUN: xdsl-opt -p convert-qref-to-qssa %s | filecheck %s
// RUN: xdsl-opt -p convert-qref-to-qssa,convert-qssa-to-qref %s | filecheck %s --check-prefix=CHECK-ROUNDTRIP

%q0, %q1 = qref.alloc<2>
qref.h %q0
qref.cz %q1, %q0
qref.cnot %q1, %q0
%0 = qref.measure %q0

// CHECK:       builtin.module {
// CHECK-NEXT:    %q0, %q1 = qssa.alloc<2>
// CHECK-NEXT:    %q0_1 = qssa.h %q0
// CHECK-NEXT:    %q1_1, %q0_2 = qssa.cz %q1, %q0_1
// CHECK-NEXT:    %q1_2, %q0_3 = qssa.cnot %q1_1, %q0_2
// CHECK-NEXT:    %0 = qssa.measure %q0_3
// CHECK-NEXT:  }

// CHECK-ROUNDTRIP:       builtin.module {
// CHECK-ROUNDTRIP-NEXT:    %q0, %q1 = qref.alloc<2>
// CHECK-ROUNDTRIP-NEXT:    qref.h %q0
// CHECK-ROUNDTRIP-NEXT:    qref.cz %q1, %q0
// CHECK-ROUNDTRIP-NEXT:    qref.cnot %q1, %q0
// CHECK-ROUNDTRIP-NEXT:    %0 = qref.measure %q0
// CHECK-ROUNDTRIP-NEXT:  }

