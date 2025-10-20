// RUN: MLIR_GENERIC_ROUNDTRIP
// RUN: MLIR_ROUNDTRIP

%arg0, %arg1, %f1 = "test.op"() : () -> (vector<8xi32>, vector<8xi32>, vector<8xf32>)
// CHECK: [[arg0:%\d+]], [[arg1:%\d+]], [[f1:%\d+]]

%add = llvm.add %arg0, %arg1 : vector<8xi32>
// CHECK: llvm.add [[arg0]], [[arg1]] : vector<8xi32>

%add2 = llvm.add %arg0, %arg1 {"nsw"} : vector<8xi32>
// CHECK: llvm.add [[arg0]], [[arg1]] {nsw} : vector<8xi32>

%sub = llvm.sub %arg0, %arg1 : vector<8xi32>
// CHECK: llvm.sub [[arg0]], [[arg1]] : vector<8xi32>

%mul = llvm.mul %arg0, %arg1 : vector<8xi32>
// CHECK: llvm.mul [[arg0]], [[arg1]] : vector<8xi32>

%udiv = llvm.udiv %arg0, %arg1 : vector<8xi32>
// CHECK: llvm.udiv [[arg0]], [[arg1]] : vector<8xi32>

%sdiv = llvm.sdiv %arg0, %arg1 : vector<8xi32>
// CHECK: llvm.sdiv [[arg0]], [[arg1]] : vector<8xi32>

%urem = llvm.urem %arg0, %arg1 : vector<8xi32>
// CHECK: llvm.urem [[arg0]], [[arg1]] : vector<8xi32>

%srem = llvm.srem %arg0, %arg1 : vector<8xi32>
// CHECK: llvm.srem [[arg0]], [[arg1]] : vector<8xi32>

%and = llvm.and %arg0, %arg1 : vector<8xi32>
// CHECK: llvm.and [[arg0]], [[arg1]] : vector<8xi32>

%or = llvm.or %arg0, %arg1 : vector<8xi32>
// CHECK: llvm.or [[arg0]], [[arg1]] : vector<8xi32>

%xor = llvm.xor %arg0, %arg1 : vector<8xi32>
// CHECK: llvm.xor [[arg0]], [[arg1]] : vector<8xi32>

%shl = llvm.shl %arg0, %arg1 : vector<8xi32>
// CHECK: llvm.shl [[arg0]], [[arg1]] : vector<8xi32>

%lshr = llvm.lshr %arg0, %arg1 : vector<8xi32>
// CHECK: llvm.lshr [[arg0]], [[arg1]] : vector<8xi32>

%ashr = llvm.ashr %arg0, %arg1 : vector<8xi32>
// CHECK: llvm.ashr [[arg0]], [[arg1]] : vector<8xi32>

// float arith:

%fmul = llvm.fmul %f1, %f1 : vector<8xf32>
// CHECK: llvm.fmul [[f1]], [[f1]] : vector<8xf32>

%fmul_fast = llvm.fmul %f1, %f1 {test = true, fastmathFlags = #llvm.fastmath<fast>} : vector<8xf32>
// CHECK: llvm.fmul [[f1]], [[f1]] {test = true, fastmathFlags = #llvm.fastmath<fast>} : vector<8xf32>

%fdiv = llvm.fdiv %f1, %f1 : vector<8xf32>
// CHECK: llvm.fdiv [[f1]], [[f1]] : vector<8xf32>

%fadd = llvm.fadd %f1, %f1 : vector<8xf32>
// CHECK: llvm.fadd [[f1]], [[f1]] : vector<8xf32>

%fsub = llvm.fsub %f1, %f1 : vector<8xf32>
// CHECK: llvm.fsub [[f1]], [[f1]] : vector<8xf32>

%frem = llvm.frem %f1, %f1 : vector<8xf32>
// CHECK: llvm.frem [[f1]], [[f1]] : vector<8xf32>
