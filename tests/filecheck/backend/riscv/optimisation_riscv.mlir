// RUN: xdsl-opt -p canonicalize %s | filecheck %s

builtin.module {
  %i0, %i1, %i2 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<>)
  %o0 = riscv.mv %i0 : (!riscv.reg<a0>) -> !riscv.reg<a0>
  %o1 = riscv.mv %i1 : (!riscv.reg<a1>) -> !riscv.reg<a2>
  %o2 = riscv.mv %i2 : (!riscv.reg<>) -> !riscv.reg<>

  %f0, %f1, %f2 = "test.op"() : () -> (!riscv.freg<fa0>, !riscv.freg<fa1>, !riscv.freg<>)
  %fo0 = riscv.fmv %f0 : (!riscv.freg<fa0>) -> !riscv.freg<fa0>
  %fo1 = riscv.fmv %f1 : (!riscv.freg<fa1>) -> !riscv.freg<fa2>
  %fo2 = riscv.fmv %f2 : (!riscv.freg<>) -> !riscv.freg<>
}

// CHECK: builtin.module {
// CHECK-NEXT:   %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<>)
// CHECK-NEXT:   %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<a1>) -> !riscv.reg<a2>
// CHECK-NEXT:   %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (!riscv.freg<fa0>, !riscv.freg<fa1>, !riscv.freg<>)
// CHECK-NEXT:   %{{.*}} = riscv.fmv %{{.*}} : (!riscv.freg<fa1>) -> !riscv.freg<fa2>
// CHECK-NEXT:   %{{.*}} = riscv.fmv %{{.*}} : (!riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT: }
