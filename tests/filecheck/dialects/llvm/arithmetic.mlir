// RUN: XDSL_ROUNDTRIP

%arg0, %arg1 = "test.op"() : () -> (i32, i32)

%add_both = llvm.add %arg0, %arg1 {"overflowFlags" = #llvm.overflow<nsw, nuw>} : i32
// CHECK: %add_both = llvm.add %arg0, %arg1 {"overflowFlags" = #llvm.overflow<nsw,nuw>} : i32

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

%urem = llvm.urem %arg0, %arg1 : i32
// CHECK: %urem = llvm.urem %arg0, %arg1 : i32

%srem = llvm.srem %arg0, %arg1 : i32
// CHECK: %srem = llvm.srem %arg0, %arg1 : i32

%and = llvm.and %arg0, %arg1 : i32
// CHECK: %and = llvm.and %arg0, %arg1 : i32

%or = llvm.or %arg0, %arg1 : i32
// CHECK: %or = llvm.or %arg0, %arg1 : i32

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
