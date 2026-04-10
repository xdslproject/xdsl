// RUN: xdsl-opt --split-input-file --verify-diagnostics %s | filecheck %s

%0 = "test.op"(): () -> !riscv.reg

%1 = riscv_stack.alloca : () -> !riscv_stack.ptr<i32>
// CHECK: %{{.*}} = riscv_stack.alloca : () -> !riscv_stack.ptr<i32>

riscv_stack.store %1, %0: (!riscv_stack.ptr<i32>, !riscv.reg) -> ()
// CHECK-NEXT: riscv_stack.store %{{.*}}, %{{.*}} : (!riscv_stack.ptr<i32>, !riscv.reg) -> ()

riscv_stack.load %1: (!riscv_stack.ptr<i32>) -> !riscv.reg
// CHECK-NEXT: %{{.*}} = riscv_stack.load %{{.*}} : (!riscv_stack.ptr<i32>) -> !riscv.reg
