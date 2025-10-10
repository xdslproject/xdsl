// RUN: MLIR_GENERIC_ROUNDTRIP
// RUN: MLIR_ROUNDTRIP

builtin.module {
    %arg0, %arg1, %f1 = "test.op"() : () -> (i32, i32, f32)
    // CHECK: [[arg0:%\d+]], [[arg1:%\d+]], [[f1:%\d+]]

    %add = llvm.add %arg0, %arg1 : i32
    // CHECK: llvm.add [[arg0]], [[arg1]] : i32

    %add2 = llvm.add %arg0, %arg1 {"nsw"} : i32
    // CHECK: llvm.add [[arg0]], [[arg1]] {nsw} : i32

    %sub = llvm.sub %arg0, %arg1 : i32
    // CHECK: llvm.sub [[arg0]], [[arg1]] : i32

    %mul = llvm.mul %arg0, %arg1 : i32
    // CHECK: llvm.mul [[arg0]], [[arg1]] : i32

    %udiv = llvm.udiv %arg0, %arg1 : i32
    // CHECK: llvm.udiv [[arg0]], [[arg1]] : i32

    %sdiv = llvm.sdiv %arg0, %arg1 : i32
    // CHECK: llvm.sdiv [[arg0]], [[arg1]] : i32

    %urem = llvm.urem %arg0, %arg1 : i32
    // CHECK: llvm.urem [[arg0]], [[arg1]] : i32

    %srem = llvm.srem %arg0, %arg1 : i32
    // CHECK: llvm.srem [[arg0]], [[arg1]] : i32

    %and = llvm.and %arg0, %arg1 : i32
    // CHECK: llvm.and [[arg0]], [[arg1]] : i32

    %or = llvm.or %arg0, %arg1 : i32
    // CHECK: llvm.or [[arg0]], [[arg1]] : i32

    %xor = llvm.xor %arg0, %arg1 : i32
    // CHECK: llvm.xor [[arg0]], [[arg1]] : i32

    %shl = llvm.shl %arg0, %arg1 : i32
    // CHECK: llvm.shl [[arg0]], [[arg1]] : i32

    %lshr = llvm.lshr %arg0, %arg1 : i32
    // CHECK: llvm.lshr [[arg0]], [[arg1]] : i32

    %ashr = llvm.ashr %arg0, %arg1 : i32
    // CHECK: llvm.ashr [[arg0]], [[arg1]] : i32

    // float arith:

    %fmul = llvm.fmul %f1, %f1 : f32
    // CHECK: llvm.fmul [[f1]], [[f1]] : f32

    %fmul_fast = llvm.fmul %f1, %f1 {test = true, fastmathFlags = #llvm.fastmath<fast>} : f32
    // CHECK: llvm.fmul [[f1]], [[f1]] {test = true, fastmathFlags = #llvm.fastmath<fast>} : f32

    %fdiv = llvm.fdiv %f1, %f1 : f32
    // CHECK: llvm.fdiv [[f1]], [[f1]] : f32

    %fadd = llvm.fadd %f1, %f1 : f32
    // CHECK: llvm.fadd [[f1]], [[f1]] : f32

    %fsub = llvm.fsub %f1, %f1 : f32
    // CHECK: llvm.fsub [[f1]], [[f1]] : f32

    %frem = llvm.frem %f1, %f1 : f32
    // CHECK: llvm.frem [[f1]], [[f1]] : f32
}
