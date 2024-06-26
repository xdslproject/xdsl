// RUN: xdsl-opt -p convert-qssa-to-qref %s | filecheck %s
// RUN: xdsl-opt -p convert-qssa-to-qref,convert-qref-to-qssa %s | filecheck %s --check-prefix=CHECK-ROUNDTRIP

%q0, %q1 = qssa.alloc<2>
%q2 = qssa.h %q0
%q3, %q4 = qssa.cz %q1, %q2
%q5, %q6 = qssa.cnot %q3, %q4
%0 = qssa.measure %q6

// CHECK:       builtin.module {
// CHECK-NEXT:    %q0, %q1 = qref.alloc<2>
// CHECK-NEXT:    qref.h %q0
// CHECK-NEXT:    qref.cz %q1, %q0
// CHECK-NEXT:    qref.cnot %q1, %q0
// CHECK-NEXT:    %0 = qref.measure %q0
// CHECK-NEXT:  }

// CHECK-ROUNDTRIP:       builtin.module {
// CHECK-ROUNDTRIP-NEXT:    %q0, %q1 = qssa.alloc<2>
// CHECK-ROUNDTRIP-NEXT:    %q0_1 = qssa.h %q0
// CHECK-ROUNDTRIP-NEXT:    %q1_1, %q0_2 = qssa.cz %q1, %q0_1
// CHECK-ROUNDTRIP-NEXT:    %q1_2, %q0_3 = qssa.cnot %q1_1, %q0_2
// CHECK-ROUNDTRIP-NEXT:    %0 = qssa.measure %q0_3
// CHECK-ROUNDTRIP-NEXT:  }

