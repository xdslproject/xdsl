// RUN: xdsl-opt -p convert-qref-to-qssa %s | filecheck %s
// RUN: xdsl-opt -p convert-qref-to-qssa,convert-qssa-to-qref %s | filecheck %s --check-prefix=CHECK-ROUNDTRIP

%q0, %q1 = qref.alloc<2>
qref.h %q0
qref.rz <pi:2> %q1
qref.cnot %q0, %q1
%0 = qref.measure %q0

// CHECK:       builtin.module {
// CHECK-NEXT:    %q0, %q1 = qssa.alloc<2>
// CHECK-NEXT:    %q0_1 = qssa.h %q0
// CHECK-NEXT:    %q1_1 = qssa.rz <pi:2> %q1
// CHECK-NEXT:    %q0_2, %q1_2 = qssa.cnot %q0_1, %q1_1
// CHECK-NEXT:    %0 = qssa.measure %q0_2
// CHECK-NEXT:  }

// CHECK-ROUNDTRIP:       builtin.module {
// CHECK-ROUNDTRIP-NEXT:    %q0, %q1 = qref.alloc<2>
// CHECK-ROUNDTRIP-NEXT:    qref.h %q0
// CHECK-ROUNDTRIP-NEXT:    qref.rz <pi:2> %q1
// CHECK-ROUNDTRIP-NEXT:    qref.cnot %q0, %q1
// CHECK-ROUNDTRIP-NEXT:    %0 = qref.measure %q0
// CHECK-ROUNDTRIP-NEXT:  }

