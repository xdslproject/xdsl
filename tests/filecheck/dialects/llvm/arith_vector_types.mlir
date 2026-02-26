// RUN: XDSL_ROUNDTRIP

%varg0, %varg1, %vf1 = "test.op"() : () -> (vector<8xi32>, vector<8xi32>, vector<8xf32>)

%vadd_both = llvm.add %varg0, %varg1 {"overflowFlags" = #llvm.overflow<nsw, nuw>} : vector<8xi32>
// CHECK: %vadd_both = llvm.add %varg0, %varg1 {overflowFlags = #llvm.overflow<nsw,nuw>} : vector<8xi32>

%vadd_both_pretty = llvm.add %varg0, %varg1 overflow<nsw, nuw> : vector<8xi32>
// CHECK: %vadd_both_pretty = llvm.add %varg0, %varg1 overflow<nsw,nuw> : vector<8xi32>

%vsub = llvm.sub %varg0, %varg1 : vector<8xi32>
// CHECK: %vsub = llvm.sub %varg0, %varg1 : vector<8xi32>

%vsub_overflow = llvm.sub %varg0, %varg1 overflow<nsw> : vector<8xi32>
// CHECK: %vsub_overflow = llvm.sub %varg0, %varg1 overflow<nsw> : vector<8xi32>

%vmul = llvm.mul %varg0, %varg1 : vector<8xi32>
// CHECK: %vmul = llvm.mul %varg0, %varg1 : vector<8xi32>

%vmul_overflow = llvm.mul %varg0, %varg1 overflow<nsw> : vector<8xi32>
// CHECK: %vmul_overflow = llvm.mul %varg0, %varg1 overflow<nsw> : vector<8xi32>

%vudiv = llvm.udiv %varg0, %varg1 : vector<8xi32>
// CHECK: %vudiv = llvm.udiv %varg0, %varg1 : vector<8xi32>

%vsdiv = llvm.sdiv %varg0, %varg1 : vector<8xi32>
// CHECK: %vsdiv = llvm.sdiv %varg0, %varg1 : vector<8xi32>

%vudiv_exact = llvm.udiv exact %varg0, %varg1 : vector<8xi32>
// CHECK: %vudiv_exact = llvm.udiv exact %varg0, %varg1 : vector<8xi32>

%vsdiv_exact = llvm.sdiv exact %varg0, %varg1 : vector<8xi32>
// CHECK: %vsdiv_exact = llvm.sdiv exact %varg0, %varg1 : vector<8xi32>

%vurem = llvm.urem %varg0, %varg1 : vector<8xi32>
// CHECK: %vurem = llvm.urem %varg0, %varg1 : vector<8xi32>

%vsrem = llvm.srem %varg0, %varg1 : vector<8xi32>
// CHECK: %vsrem = llvm.srem %varg0, %varg1 : vector<8xi32>

%vand = llvm.and %varg0, %varg1 : vector<8xi32>
// CHECK: %vand = llvm.and %varg0, %varg1 : vector<8xi32>

%vor = llvm.or %varg0, %varg1 : vector<8xi32>
// CHECK: %vor = llvm.or %varg0, %varg1 : vector<8xi32>

%vor_disjoint = llvm.or disjoint %varg0, %varg1 : vector<8xi32>
// CHECK: %vor_disjoint = llvm.or disjoint %varg0, %varg1 : vector<8xi32>

%vxor = llvm.xor %varg0, %varg1 : vector<8xi32>
// CHECK: %vxor = llvm.xor %varg0, %varg1 : vector<8xi32>

%vshl = llvm.shl %varg0, %varg1 : vector<8xi32>
// CHECK: %vshl = llvm.shl %varg0, %varg1 : vector<8xi32>

%vshl_overflow = llvm.shl %varg0, %varg1 overflow<nsw> : vector<8xi32>
// CHECK: %vshl_overflow = llvm.shl %varg0, %varg1 overflow<nsw> : vector<8xi32>

%vlshr = llvm.lshr %varg0, %varg1 : vector<8xi32>
// CHECK: %vlshr = llvm.lshr %varg0, %varg1 : vector<8xi32>

%vashr = llvm.ashr %varg0, %varg1 : vector<8xi32>
// CHECK: %vashr = llvm.ashr %varg0, %varg1 : vector<8xi32>

%vlshr_exact = llvm.lshr exact %varg0, %varg1 : vector<8xi32>
// CHECK: %vlshr_exact = llvm.lshr exact %varg0, %varg1 : vector<8xi32>

%vashr_exact = llvm.ashr exact %varg0, %varg1 : vector<8xi32>
// CHECK: %vashr_exact = llvm.ashr exact %varg0, %varg1 : vector<8xi32>

%vtrunc = llvm.trunc %varg0 : vector<8xi32> to vector<8xi16>
// CHECK: %vtrunc = llvm.trunc %varg0 : vector<8xi32> to vector<8xi16>

%vtrunc_overflow = llvm.trunc %varg0 overflow<nsw> : vector<8xi32> to vector<8xi16>
// CHECK: %vtrunc_overflow = llvm.trunc %varg0 overflow<nsw> : vector<8xi32> to vector<8xi16>

%vsext = llvm.sext %varg0 : vector<8xi32> to vector<8xi64>
// CHECK: %vsext = llvm.sext %varg0 : vector<8xi32> to vector<8xi64>

%vzext = llvm.zext %varg0 : vector<8xi32> to vector<8xi64>
// CHECK: %vzext = llvm.zext %varg0 : vector<8xi32> to vector<8xi64>

%vzext_nneg = llvm.zext nneg %varg0 : vector<8xi32> to vector<8xi64>
// CHECK: %vzext_nneg = llvm.zext nneg %varg0 : vector<8xi32> to vector<8xi64>

%vcst1 = llvm.mlir.constant(dense<false> : vector<8xi1>) : vector<8xi1>
// CHECK: %vcst1 = llvm.mlir.constant(dense<false> : vector<8xi1>) : vector<8xi1>

%vcst64 = llvm.mlir.constant(dense<25> : vector<8xi64>) : vector<8xi64>
// CHECK: %vcst64 = llvm.mlir.constant(dense<25> : vector<8xi64>) : vector<8xi64>

%vcst32 = llvm.mlir.constant(dense<25> : vector<8xi32>) : vector<8xi32>
// CHECK: %vcst32 = llvm.mlir.constant(dense<25> : vector<8xi32>) : vector<8xi32>

%vicmp_eq = llvm.icmp "eq" %varg0, %varg1 : vector<8xi32>
// CHECK: %vicmp_eq = llvm.icmp "eq" %varg0, %varg1 : vector<8xi32>

%vicmp_ne = llvm.icmp "ne" %varg0, %varg1 : vector<8xi32>
// CHECK: %vicmp_ne = llvm.icmp "ne" %varg0, %varg1 : vector<8xi32>

%vicmp_slt = llvm.icmp "slt" %varg0, %varg1 : vector<8xi32>
// CHECK: %vicmp_slt = llvm.icmp "slt" %varg0, %varg1 : vector<8xi32>

%vicmp_sle = llvm.icmp "sle" %varg0, %varg1 : vector<8xi32>
// CHECK: %vicmp_sle = llvm.icmp "sle" %varg0, %varg1 : vector<8xi32>

%vicmp_sgt = llvm.icmp "sgt" %varg0, %varg1 : vector<8xi32>
// CHECK: %vicmp_sgt = llvm.icmp "sgt" %varg0, %varg1 : vector<8xi32>

%vicmp_sge = llvm.icmp "sge" %varg0, %varg1 : vector<8xi32>
// CHECK: %vicmp_sge = llvm.icmp "sge" %varg0, %varg1 : vector<8xi32>

%vicmp_ult = llvm.icmp "ult" %varg0, %varg1 : vector<8xi32>
// CHECK: %vicmp_ult = llvm.icmp "ult" %varg0, %varg1 : vector<8xi32>

%vicmp_ule = llvm.icmp "ule" %varg0, %varg1 : vector<8xi32>
// CHECK: %vicmp_ule = llvm.icmp "ule" %varg0, %varg1 : vector<8xi32>

%vicmp_ugt = llvm.icmp "ugt" %varg0, %varg1 : vector<8xi32>
// CHECK: %vicmp_ugt = llvm.icmp "ugt" %varg0, %varg1 : vector<8xi32>

%vicmp_uge = llvm.icmp "uge" %varg0, %varg1 : vector<8xi32>
// CHECK: %vicmp_uge = llvm.icmp "uge" %varg0, %varg1 : vector<8xi32>

// float arith:

%vfmul = llvm.fmul %vf1, %vf1 : vector<8xf32>
// CHECK: %vfmul = llvm.fmul %vf1, %vf1 : vector<8xf32>

%vfmul_fast = llvm.fmul %vf1, %vf1 {fastmathFlags = #llvm.fastmath<fast>} : vector<8xf32>
// CHECK: %vfmul_fast = llvm.fmul %vf1, %vf1 {fastmathFlags = #llvm.fastmath<fast>} : vector<8xf32>

%vfdiv = llvm.fdiv %vf1, %vf1 : vector<8xf32>
// CHECK: %vfdiv = llvm.fdiv %vf1, %vf1 : vector<8xf32>

%vfadd = llvm.fadd %vf1, %vf1 : vector<8xf32>
// CHECK: %vfadd = llvm.fadd %vf1, %vf1 : vector<8xf32>

%vfsub = llvm.fsub %vf1, %vf1 : vector<8xf32>
// CHECK: %vfsub = llvm.fsub %vf1, %vf1 : vector<8xf32>

%vfrem = llvm.frem %vf1, %vf1 : vector<8xf32>
// CHECK: %vfrem = llvm.frem %vf1, %vf1 : vector<8xf32>
