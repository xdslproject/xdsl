// RUN: XDSL_ROUNDTRIP

%arg0, %arg1, %f1 = "test.op"() : () -> (i32, i32, f32)

%add_both = llvm.add %arg0, %arg1 {"overflowFlags" = #llvm.overflow<nsw, nuw>} : i32
// CHECK: %add_both = llvm.add %arg0, %arg1 {overflowFlags = #llvm.overflow<nsw,nuw>} : i32

%add_both_pretty = llvm.add %arg0, %arg1 overflow<nsw, nuw> : i32
// CHECK: %add_both_pretty = llvm.add %arg0, %arg1 overflow<nsw,nuw> : i32

%sub = llvm.sub %arg0, %arg1 : i32
// CHECK: %sub = llvm.sub %arg0, %arg1 : i32

%sub_overflow = llvm.sub %arg0, %arg1 overflow<nsw> : i32
// CHECK: %sub_overflow = llvm.sub %arg0, %arg1 overflow<nsw> : i32

%mul = llvm.mul %arg0, %arg1 : i32
// CHECK: %mul = llvm.mul %arg0, %arg1 : i32

%mul_overflow = llvm.mul %arg0, %arg1 overflow<nsw> : i32
// CHECK: %mul_overflow = llvm.mul %arg0, %arg1 overflow<nsw> : i32

%udiv = llvm.udiv %arg0, %arg1 : i32
// CHECK: %udiv = llvm.udiv %arg0, %arg1 : i32

%sdiv = llvm.sdiv %arg0, %arg1 : i32
// CHECK: %sdiv = llvm.sdiv %arg0, %arg1 : i32

%udiv_exact = llvm.udiv exact %arg0, %arg1 : i32
// CHECK: %udiv_exact = llvm.udiv exact %arg0, %arg1 : i32

%sdiv_exact = llvm.sdiv exact %arg0, %arg1 : i32
// CHECK: %sdiv_exact = llvm.sdiv exact %arg0, %arg1 : i32

%urem = llvm.urem %arg0, %arg1 : i32
// CHECK: %urem = llvm.urem %arg0, %arg1 : i32

%srem = llvm.srem %arg0, %arg1 : i32
// CHECK: %srem = llvm.srem %arg0, %arg1 : i32

%and = llvm.and %arg0, %arg1 : i32
// CHECK: %and = llvm.and %arg0, %arg1 : i32

%or = llvm.or %arg0, %arg1 : i32
// CHECK: %or = llvm.or %arg0, %arg1 : i32

%or_disjoint = llvm.or disjoint %arg0, %arg1 : i32
// CHECK: %or_disjoint = llvm.or disjoint %arg0, %arg1 : i32

%xor = llvm.xor %arg0, %arg1 : i32
// CHECK: %xor = llvm.xor %arg0, %arg1 : i32

%shl = llvm.shl %arg0, %arg1 : i32
// CHECK: %shl = llvm.shl %arg0, %arg1 : i32

%shl_overflow = llvm.shl %arg0, %arg1 overflow<nsw> : i32
// CHECK: %shl_overflow = llvm.shl %arg0, %arg1 overflow<nsw> : i32

%lshr = llvm.lshr %arg0, %arg1 : i32
// CHECK: %lshr = llvm.lshr %arg0, %arg1 : i32

%ashr = llvm.ashr %arg0, %arg1 : i32
// CHECK: %ashr = llvm.ashr %arg0, %arg1 : i32

%lshr_exact = llvm.lshr exact %arg0, %arg1 : i32
// CHECK: %lshr_exact = llvm.lshr exact %arg0, %arg1 : i32

%ashr_exact = llvm.ashr exact %arg0, %arg1 : i32
// CHECK: %ashr_exact = llvm.ashr exact %arg0, %arg1 : i32

%trunc = llvm.trunc %arg0 : i32 to i16
// CHECK: %trunc = llvm.trunc %arg0 : i32 to i16

%trunc_overflow = llvm.trunc %arg0 overflow<nsw> : i32 to i16
// CHECK: %trunc_overflow = llvm.trunc %arg0 overflow<nsw> : i32 to i16

%sext = llvm.sext %arg0 : i32 to i64
// CHECK: %sext = llvm.sext %arg0 : i32 to i64

%zext = llvm.zext %arg0 : i32 to i64
// CHECK: %zext = llvm.zext %arg0 : i32 to i64

%zext_nneg = llvm.zext nneg %arg0 : i32 to i64
// CHECK: %zext_nneg = llvm.zext nneg %arg0 : i32 to i64

%cst1 = llvm.mlir.constant(false) : i1
// CHECK: %cst1 = llvm.mlir.constant(false) : i1

%cst64 = llvm.mlir.constant(25) : i64
// CHECK: %cst64 = llvm.mlir.constant(25) : i64

%cst32 = llvm.mlir.constant(25 : i32) : i32
// CHECK: %cst32 = llvm.mlir.constant(25 : i32) : i32

%icmp_eq = llvm.icmp "eq" %arg0, %arg1 : i32
// CHECK: %icmp_eq = llvm.icmp "eq" %arg0, %arg1 : i32

%icmp_ne = llvm.icmp "ne" %arg0, %arg1 : i32
// CHECK: %icmp_ne = llvm.icmp "ne" %arg0, %arg1 : i32

%icmp_slt = llvm.icmp "slt" %arg0, %arg1 : i32
// CHECK: %icmp_slt = llvm.icmp "slt" %arg0, %arg1 : i32

%icmp_sle = llvm.icmp "sle" %arg0, %arg1 : i32
// CHECK: %icmp_sle = llvm.icmp "sle" %arg0, %arg1 : i32

%icmp_sgt = llvm.icmp "sgt" %arg0, %arg1 : i32
// CHECK: %icmp_sgt = llvm.icmp "sgt" %arg0, %arg1 : i32

%icmp_sge = llvm.icmp "sge" %arg0, %arg1 : i32
// CHECK: %icmp_sge = llvm.icmp "sge" %arg0, %arg1 : i32

%icmp_ult = llvm.icmp "ult" %arg0, %arg1 : i32
// CHECK: %icmp_ult = llvm.icmp "ult" %arg0, %arg1 : i32

%icmp_ule = llvm.icmp "ule" %arg0, %arg1 : i32
// CHECK: %icmp_ule = llvm.icmp "ule" %arg0, %arg1 : i32

%icmp_ugt = llvm.icmp "ugt" %arg0, %arg1 : i32
// CHECK: %icmp_ugt = llvm.icmp "ugt" %arg0, %arg1 : i32

%icmp_uge = llvm.icmp "uge" %arg0, %arg1 : i32
// CHECK: %icmp_uge = llvm.icmp "uge" %arg0, %arg1 : i32

// float arith:

%fmul = llvm.fmul %f1, %f1 : f32
// CHECK: %fmul = llvm.fmul %f1, %f1 : f32

%fmul_fast = llvm.fmul %f1, %f1 {fastmathFlags = #llvm.fastmath<fast>} : f32
// CHECK: %fmul_fast = llvm.fmul %f1, %f1 {fastmathFlags = #llvm.fastmath<fast>} : f32

%fdiv = llvm.fdiv %f1, %f1 : f32
// CHECK: %fdiv = llvm.fdiv %f1, %f1 : f32

%fadd = llvm.fadd %f1, %f1 : f32
// CHECK: %fadd = llvm.fadd %f1, %f1 : f32

%fsub = llvm.fsub %f1, %f1 : f32
// CHECK: %fsub = llvm.fsub %f1, %f1 : f32

%frem = llvm.frem %f1, %f1 : f32
// CHECK: %frem = llvm.frem %f1, %f1 : f32
