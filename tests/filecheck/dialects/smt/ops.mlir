// RUN: XDSL_ROUNDTRIP

%arg1 = smt.constant true
%arg2 = smt.constant false
%arg3 = smt.constant false

%and = smt.or %arg1, %arg2, %arg3
%or = smt.or %arg1, %arg2, %arg3
%xor = smt.or %arg1, %arg2, %arg3


// CHECK:         %arg1 = smt.constant true
// CHECK-NEXT:    %arg2 = smt.constant false
// CHECK-NEXT:    %arg3 = smt.constant false
// CHECK-NEXT:    %and = smt.or %arg1, %arg2, %arg3
// CHECK-NEXT:    %or = smt.or %arg1, %arg2, %arg3
// CHECK-NEXT:    %xor = smt.or %arg1, %arg2, %arg3
