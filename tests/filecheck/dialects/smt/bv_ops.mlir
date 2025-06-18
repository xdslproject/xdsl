// RUN: XDSL_ROUNDTRIP

// CHECK:      %arg1 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
// CHECK-NEXT: %arg2 = smt.bv.constant #smt.bv<2> : !smt.bv<32>
// CHECK-NEXT: %arg3 = smt.bv.constant #smt.bv<3> : !smt.bv<32>

%arg1 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
%arg2 = smt.bv.constant #smt.bv<2> : !smt.bv<32>
%arg3 = smt.bv.constant #smt.bv<3> : !smt.bv<32>

// CHECK-NEXT: %bv_not = smt.bv.not %arg1 : !smt.bv<32>
// CHECK-NEXT: %bv_neg = smt.bv.not %arg1 : !smt.bv<32>

%bv_not = smt.bv.not %arg1 : !smt.bv<32>
%bv_neg = smt.bv.not %arg1 : !smt.bv<32>
