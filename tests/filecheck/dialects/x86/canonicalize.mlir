// RUN: xdsl-opt -p canonicalize %s | filecheck %s

// CHECK:       builtin.module {

// CHECK-NEXT:    %i0, %i1, %i2 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg)
// CHECK-NEXT:    %o1 = riscv.mv %i1 : (!riscv.reg<a1>) -> !riscv.reg<a2>
// CHECK-NEXT:    %o2 = riscv.mv %i2 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:    "test.op"(%i0, %o1, %o2) : (!riscv.reg<a0>, !riscv.reg<a2>, !riscv.reg) -> ()

%i0, %i1, %i2 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg)
%o0 = riscv.mv %i0 : (!riscv.reg<a0>) -> !riscv.reg<a0>
%o1 = riscv.mv %i1 : (!riscv.reg<a1>) -> !riscv.reg<a2>
%o2 = riscv.mv %i2 : (!riscv.reg) -> !riscv.reg
"test.op"(%o0, %o1, %o2) : (!riscv.reg<a0>, !riscv.reg<a2>, !riscv.reg) -> ()

// CHECK-NEXT:  }
