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

// CHECK-NEXT: %bv_and = smt.bv.and %arg1, %arg2 : !smt.bv<32>
// CHECK-NEXT: %bv_or = smt.bv.or %arg1, %arg2 : !smt.bv<32>
// CHECK-NEXT: %bv_xor = smt.bv.xor %arg1, %arg2 : !smt.bv<32>
// CHECK-NEXT: %bv_add = smt.bv.add %arg1, %arg2 : !smt.bv<32>
// CHECK-NEXT: %bv_mul = smt.bv.mul %arg1, %arg2 : !smt.bv<32>
// CHECK-NEXT: %bv_udiv = smt.bv.udiv %arg1, %arg2 : !smt.bv<32>
// CHECK-NEXT: %bv_sdiv = smt.bv.sdiv %arg1, %arg2 : !smt.bv<32>
// CHECK-NEXT: %bv_urem = smt.bv.urem %arg1, %arg2 : !smt.bv<32>
// CHECK-NEXT: %bv_srem = smt.bv.srem %arg1, %arg2 : !smt.bv<32>
// CHECK-NEXT: %bv_smod = smt.bv.smod %arg1, %arg2 : !smt.bv<32>
// CHECK-NEXT: %bv_shl = smt.bv.shl %arg1, %arg2 : !smt.bv<32>
// CHECK-NEXT: %bv_lshr = smt.bv.lshr %arg1, %arg2 : !smt.bv<32>
// CHECK-NEXT: %bv_ashr = smt.bv.ashr %arg1, %arg2 : !smt.bv<32>

%bv_and = smt.bv.and %arg1, %arg2 : !smt.bv<32>
%bv_or = smt.bv.or %arg1, %arg2 : !smt.bv<32>
%bv_xor = smt.bv.xor %arg1, %arg2 : !smt.bv<32>
%bv_add = smt.bv.add %arg1, %arg2 : !smt.bv<32>
%bv_mul = smt.bv.mul %arg1, %arg2 : !smt.bv<32>
%bv_udiv = smt.bv.udiv %arg1, %arg2 : !smt.bv<32>
%bv_sdiv = smt.bv.sdiv %arg1, %arg2 : !smt.bv<32>
%bv_urem = smt.bv.urem %arg1, %arg2 : !smt.bv<32>
%bv_srem = smt.bv.srem %arg1, %arg2 : !smt.bv<32>
%bv_smod = smt.bv.smod %arg1, %arg2 : !smt.bv<32>
%bv_shl = smt.bv.shl %arg1, %arg2 : !smt.bv<32>
%bv_lshr = smt.bv.lshr %arg1, %arg2 : !smt.bv<32>
%bv_ashr = smt.bv.ashr %arg1, %arg2 : !smt.bv<32>
