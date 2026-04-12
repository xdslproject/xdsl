// RUN: xdsl-opt --split-input-file -p convert-riscv-stack-to-riscv %s | filecheck %s

builtin.module {
  riscv_func.func @main() {
    %0 = "test.op"() : () -> !riscv.reg
    %1 = riscv_stack.alloca : () -> !riscv_stack.ptr<i32>
    riscv_stack.store %1, %0 : (!riscv_stack.ptr<i32>, !riscv.reg) -> ()
    %2 = riscv_stack.load %1 : (!riscv_stack.ptr<i32>) -> !riscv.reg
    %3 = riscv_stack.alloca : () -> !riscv_stack.ptr<i32>
    riscv_stack.store %3, %2 : (!riscv_stack.ptr<i32>, !riscv.reg) -> ()
    riscv_func.return
  }
}
// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @main() {
// CHECK-NEXT:      %0 = rv32.get_register : !riscv.reg<sp>
// CHECK-NEXT:      %1 = riscv.addi %0, -8 : (!riscv.reg<sp>) -> !riscv.reg<sp>
// CHECK-NEXT:      %2 = "test.op"() : () -> !riscv.reg
// CHECK-NEXT:      riscv.sw %0, %2, 0 : (!riscv.reg<sp>, !riscv.reg) -> ()
// CHECK-NEXT:      %3 = riscv.lw %0, 0 : (!riscv.reg<sp>) -> !riscv.reg
// CHECK-NEXT:      riscv.sw %0, %3, 4 : (!riscv.reg<sp>, !riscv.reg) -> ()
// CHECK-NEXT:      %4 = riscv.addi %0, 8 : (!riscv.reg<sp>) -> !riscv.reg<sp>
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----
// Test floats
builtin.module {
  riscv_func.func @main() {
    %0, %1 = "test.op"() : () -> (!riscv.reg, !riscv.freg)
    %3 = riscv_stack.alloca : () -> !riscv_stack.ptr<f64>
    %2 = riscv_stack.alloca : () -> !riscv_stack.ptr<i32>

    riscv_stack.store %2, %0 : (!riscv_stack.ptr<i32>, !riscv.reg) -> ()
    riscv_stack.store %3, %1 : (!riscv_stack.ptr<f64>, !riscv.freg) -> ()

    %4 = riscv_stack.load %2 : (!riscv_stack.ptr<i32>) -> (!riscv.reg)
    %5 = riscv_stack.load %3 : (!riscv_stack.ptr<f64>) -> (!riscv.freg)

    riscv_func.return
  }
}
// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @main() {
// CHECK-NEXT:      %0 = rv32.get_register : !riscv.reg<sp>
// CHECK-NEXT:      %1 = riscv.addi %0, -12 : (!riscv.reg<sp>) -> !riscv.reg<sp>
// CHECK-NEXT:      %2, %3 = "test.op"() : () -> (!riscv.reg, !riscv.freg)
// CHECK-NEXT:      riscv.sw %0, %2, 8 : (!riscv.reg<sp>, !riscv.reg) -> ()
// CHECK-NEXT:      riscv.fsd %0, %3, 0 : (!riscv.reg<sp>, !riscv.freg) -> ()
// CHECK-NEXT:      %4 = riscv.lw %0, 8 : (!riscv.reg<sp>) -> !riscv.reg
// CHECK-NEXT:      %5 = riscv.fld %0, 0 : (!riscv.reg<sp>) -> !riscv.freg
// CHECK-NEXT:      %6 = riscv.addi %0, 12 : (!riscv.reg<sp>) -> !riscv.reg<sp>
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
