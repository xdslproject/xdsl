// RUN: xdsl-opt -p convert-memref-to-riscv %s --split-input-file --verify-diagnostics | filecheck %s

builtin.module {
    %v, %r, %c, %m = "test.op"() : () -> (f32, index, index, memref<3x2xf32>)
    "memref.store"(%v, %m, %r, %c) {"nontemporal" = false} : (f32, memref<3x2xf32>, index, index) -> ()
    %x = "memref.load"(%m, %r, %c) {"nontemporal" = false} : (memref<3x2xf32>, index, index) -> (f32)
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %v, %r, %c, %m = "test.op"() : () -> (f32, index, index, memref<3x2xf32>)
// CHECK-NEXT:   %0 = builtin.unrealized_conversion_cast %v : f32 to !riscv.freg<>
// CHECK-NEXT:   %1 = builtin.unrealized_conversion_cast %m : memref<3x2xf32> to !riscv.reg<>
// CHECK-NEXT:   %2 = builtin.unrealized_conversion_cast %r : index to !riscv.reg<>
// CHECK-NEXT:   %3 = builtin.unrealized_conversion_cast %c : index to !riscv.reg<>
// CHECK-NEXT:   %4 = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:   %5 = riscv.mul %4, %2 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %6 = riscv.add %5, %3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %7 = riscv.slli %6, 2 {"comment" = "mutiply by elm size"} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %8 = riscv.add %1, %7 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   riscv.fsw %8, %0, 0 {"comment" = "store float value to memref of shape (3, 2)"} : (!riscv.reg<>, !riscv.freg<>) -> ()
// CHECK-NEXT:   %9 = builtin.unrealized_conversion_cast %m : memref<3x2xf32> to !riscv.reg<>
// CHECK-NEXT:   %10 = builtin.unrealized_conversion_cast %r : index to !riscv.reg<>
// CHECK-NEXT:   %11 = builtin.unrealized_conversion_cast %c : index to !riscv.reg<>
// CHECK-NEXT:   %12 = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:   %13 = riscv.mul %12, %10 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %14 = riscv.add %13, %11 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %15 = riscv.slli %14, 2 {"comment" = "mutiply by elm size"} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %16 = riscv.add %9, %15 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %x = riscv.flw %16, 0 {"comment" = "load value from memref of shape (3, 2)"} : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-NEXT:   %x_1 = builtin.unrealized_conversion_cast %x : !riscv.freg<> to f32
// CHECK-NEXT: }

// -----

builtin.module {
    %m = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<1x1xf32>
}

// CHECK:      Lowering memref.alloc not implemented yet

// -----

builtin.module {
    %m = "test.op"() : () -> memref<1x1xf32>
    "memref.dealloc"(%m) : (memref<1x1xf32>) -> ()
}

// CHECK:      Lowering memref.dealloc not implemented yet
