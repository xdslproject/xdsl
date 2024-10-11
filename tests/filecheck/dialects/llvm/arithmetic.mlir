// RUN: XDSL_ROUNDTRIP

builtin.module {
    %arg0, %arg1 = "test.op"() : () -> (i32, i32)

    %add = llvm.add %arg0, %arg1 : i32
    // CHECK: %add = llvm.add %arg0, %arg1 : i32

    %add2 = llvm.add %arg0, %arg1 {"nsw"} : i32
    // CHECK: %add2 = llvm.add %arg0, %arg1 {"nsw"} : i32

    %sub = llvm.sub %arg0, %arg1 : i32
    // CHECK: %sub = llvm.sub %arg0, %arg1 : i32

    %mul = llvm.mul %arg0, %arg1 : i32
    // CHECK: %mul = llvm.mul %arg0, %arg1 : i32

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

    %lshr = llvm.lshr %arg0, %arg1 : i32
    // CHECK: %lshr = llvm.lshr %arg0, %arg1 : i32

    %ashr = llvm.ashr %arg0, %arg1 : i32
    // CHECK: %ashr = llvm.ashr %arg0, %arg1 : i32
}
