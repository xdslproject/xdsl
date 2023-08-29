// RUN: xdsl-opt -p convert-memref-to-riscv,reconcile-unrealized-casts,canonicalize %s  | filecheck %s

// Test that a memref store and load with constant indices optimise to a single operation

builtin.module {
    %v_reg, %m_reg = "test.op"() : () -> (!riscv.freg<>, !riscv.reg<>)
    %v = builtin.unrealized_conversion_cast %v_reg : !riscv.freg<> to f32
    %m = builtin.unrealized_conversion_cast %m_reg : !riscv.reg<> to memref<3x2xf32>
    %r_reg = riscv.li 1 : () -> !riscv.reg<>
    %c_reg = riscv.li 1 : () -> !riscv.reg<>
    %r = builtin.unrealized_conversion_cast %r_reg : !riscv.reg<> to index
    %c = builtin.unrealized_conversion_cast %c_reg : !riscv.reg<> to index
    "memref.store"(%v, %m, %r, %c) {"nontemporal" = false} : (f32, memref<3x2xf32>, index, index) -> ()
    %x = "memref.load"(%m, %r, %c) {"nontemporal" = false} : (memref<3x2xf32>, index, index) -> (f32)
    %x_reg = builtin.unrealized_conversion_cast %x : f32 to !riscv.freg<>
    "test.op"(%x_reg) : (!riscv.freg<>) -> ()
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %v_reg, %m_reg = "test.op"() : () -> (!riscv.freg<>, !riscv.reg<>)
// CHECK-NEXT:   riscv.fsw %m_reg, %v_reg, 12 {"comment" = "store float value to memref of shape (3, 2)"} : (!riscv.reg<>, !riscv.freg<>) -> ()
// CHECK-NEXT:   %x = riscv.flw %m_reg, 12 {"comment" = "load value from memref of shape (3, 2)"} : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-NEXT:   "test.op"(%x) : (!riscv.freg<>) -> ()
// CHECK-NEXT: }
