// RUN: XDSL_ROUNDTRIP

%arg1 = smt.constant true
%arg2 = smt.constant false
%arg3 = smt.constant false

%and = smt.or %arg1, %arg2, %arg3
%or = smt.and %arg1, %arg2, %arg3
%xor = smt.xor %arg1, %arg2, %arg3

%eq = smt.eq %arg1, %arg2, %arg3 : !smt.bool
%distinct = smt.distinct %arg1, %arg2, %arg3 : !smt.bool

// CHECK:         %arg1 = smt.constant true
// CHECK-NEXT:    %arg2 = smt.constant false
// CHECK-NEXT:    %arg3 = smt.constant false
// CHECK-NEXT:    %and = smt.or %arg1, %arg2, %arg3
// CHECK-NEXT:    %or = smt.and %arg1, %arg2, %arg3
// CHECK-NEXT:    %xor = smt.xor %arg1, %arg2, %arg3
// CHECK-NEXT:    %eq = smt.eq %arg1, %arg2, %arg3 : !smt.bool
// CHECK-NEXT:    %distinct = smt.distinct %arg1, %arg2, %arg3 : !smt.bool
