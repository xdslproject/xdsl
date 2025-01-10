// RUN: mlir-opt %s --mlir-print-op-generic | xdsl-opt | filecheck %s

builtin.module {
    %arg0, %arg1 = "test.op"() : () -> (i32, i32)

    %add = llvm.add %arg0, %arg1 : i32
    // CHECK: %{{.*}} = llvm.add %{{.*}}, %{{.*}} : i32

    %add2 = llvm.add %arg0, %arg1 {"nsw"} : i32
    // CHECK: %{{.*}} = llvm.add %{{.*}}, %{{.*}} {nsw} : i32

    %sub = llvm.sub %arg0, %arg1 : i32
    // CHECK: %{{.*}} = llvm.sub %{{.*}}, %{{.*}} : i32

    %mul = llvm.mul %arg0, %arg1 : i32
    // CHECK: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i32

    %udiv = llvm.udiv %arg0, %arg1 : i32
    // CHECK: %{{.*}} = llvm.udiv %{{.*}}, %{{.*}} : i32

    %sdiv = llvm.sdiv %arg0, %arg1 : i32
    // CHECK: %{{.*}} = llvm.sdiv %{{.*}}, %{{.*}} : i32

    %urem = llvm.urem %arg0, %arg1 : i32
    // CHECK: %{{.*}} = llvm.urem %{{.*}}, %{{.*}} : i32

    %srem = llvm.srem %arg0, %arg1 : i32
    // CHECK: %{{.*}} = llvm.srem %{{.*}}, %{{.*}} : i32

    %and = llvm.and %arg0, %arg1 : i32
    // CHECK: %{{.*}} = llvm.and %{{.*}}, %{{.*}} : i32

    %or = llvm.or %arg0, %arg1 : i32
    // CHECK: %{{.*}} = llvm.or %{{.*}}, %{{.*}} : i32

    %xor = llvm.xor %arg0, %arg1 : i32
    // CHECK: %{{.*}} = llvm.xor %{{.*}}, %{{.*}} : i32

    %shl = llvm.shl %arg0, %arg1 : i32
    // CHECK: %{{.*}} = llvm.shl %{{.*}}, %{{.*}} : i32

    %lshr = llvm.lshr %arg0, %arg1 : i32
    // CHECK: %{{.*}} = llvm.lshr %{{.*}}, %{{.*}} : i32

    %ashr = llvm.ashr %arg0, %arg1 : i32
    // CHECK: %{{.*}} = llvm.ashr %{{.*}}, %{{.*}} : i32
}
