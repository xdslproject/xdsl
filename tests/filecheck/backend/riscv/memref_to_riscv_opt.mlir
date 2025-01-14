// RUN: xdsl-opt --split-input-file -p convert-memref-to-riscv,reconcile-unrealized-casts,canonicalize %s | filecheck %s

// Test that a memref float store and load with constant indices optimise to a single operation

builtin.module {
    %vf_reg, %vd_reg, %vi_reg, %mf_reg, %md_reg, %mi_reg = "test.op"() : () -> (!riscv.freg, !riscv.freg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg)
    %vf = builtin.unrealized_conversion_cast %vf_reg : !riscv.freg to f32
    %vd = builtin.unrealized_conversion_cast %vd_reg : !riscv.freg to f64
    %vi = builtin.unrealized_conversion_cast %vi_reg : !riscv.reg to i32
    %mf = builtin.unrealized_conversion_cast %mf_reg : !riscv.reg to memref<3x2xf32>
    %md = builtin.unrealized_conversion_cast %md_reg : !riscv.reg to memref<3x2xf64>
    %mi = builtin.unrealized_conversion_cast %mi_reg : !riscv.reg to memref<3x2xi32>
    %r_reg = riscv.li 1 : !riscv.reg
    %c_reg = riscv.li 1 : !riscv.reg
    %r = builtin.unrealized_conversion_cast %r_reg : !riscv.reg to index
    %c = builtin.unrealized_conversion_cast %c_reg : !riscv.reg to index
    "memref.store"(%vf, %mf, %r, %c) {"nontemporal" = false} : (f32, memref<3x2xf32>, index, index) -> ()
    %xf = "memref.load"(%mf, %r, %c) {"nontemporal" = false} : (memref<3x2xf32>, index, index) -> (f32)
    "memref.store"(%vd, %md, %r, %c) {"nontemporal" = false} : (f64, memref<3x2xf64>, index, index) -> ()
    %xd = "memref.load"(%md, %r, %c) {"nontemporal" = false} : (memref<3x2xf64>, index, index) -> (f64)
    "memref.store"(%vi, %mi, %r, %c) {"nontemporal" = false} : (i32, memref<3x2xi32>, index, index) -> ()
    %xi = "memref.load"(%mi, %r, %c) {"nontemporal" = false} : (memref<3x2xi32>, index, index) -> (i32)
    %xf_reg = builtin.unrealized_conversion_cast %xf : f32 to !riscv.freg
    %xd_reg = builtin.unrealized_conversion_cast %xd : f64 to !riscv.freg
    %xi_reg = builtin.unrealized_conversion_cast %xi : i32 to !riscv.reg
    "test.op"(%xf_reg) : (!riscv.freg) -> ()
    "test.op"(%xd_reg) : (!riscv.freg) -> ()
    "test.op"(%xi_reg) : (!riscv.reg) -> ()
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %vf_reg, %vd_reg, %vi_reg, %mf_reg, %md_reg, %mi_reg = "test.op"() : () -> (!riscv.freg, !riscv.freg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg)
// CHECK-NEXT:   riscv.fsw %mf_reg, %vf_reg, 12 {comment = "store float value to memref of shape (3, 2)"} : (!riscv.reg, !riscv.freg) -> ()
// CHECK-NEXT:   %xf = riscv.flw %mf_reg, 12 {comment = "load float from memref of shape (3, 2)"} : (!riscv.reg) -> !riscv.freg
// CHECK-NEXT:   riscv.fsd %md_reg, %vd_reg, 24 {comment = "store double value to memref of shape (3, 2)"} : (!riscv.reg, !riscv.freg) -> ()
// CHECK-NEXT:   %xd = riscv.fld %md_reg, 24 {comment = "load double from memref of shape (3, 2)"} : (!riscv.reg) -> !riscv.freg
// CHECK-NEXT:   riscv.sw %mi_reg, %vi_reg, 12 {comment = "store int value to memref of shape (3, 2)"} : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:   %xi = riscv.lw %mi_reg, 12 {comment = "load word from memref of shape (3, 2)"} : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:   "test.op"(%xf) : (!riscv.freg) -> ()
// CHECK-NEXT:   "test.op"(%xd) : (!riscv.freg) -> ()
// CHECK-NEXT:   "test.op"(%xi) : (!riscv.reg) -> ()
// CHECK-NEXT: }
