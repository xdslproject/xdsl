// RUN: xdsl-opt -p convert-memref-to-riscv  --split-input-file --verify-diagnostics %s | filecheck %s

builtin.module {
    %v_f32, %v_i32, %r, %c, %m_f32, %m_i32 = "test.op"() : () -> (f32, i32, index, index, memref<3x2xf32>, memref<3xi32>)
    "memref.store"(%v_f32, %m_f32, %r, %c) {"nontemporal" = false} : (f32, memref<3x2xf32>, index, index) -> ()
    %x_f32 = "memref.load"(%m_f32, %r, %c) {"nontemporal" = false} : (memref<3x2xf32>, index, index) -> (f32)
    "memref.store"(%v_i32, %m_i32, %c) {"nontemporal" = false} : (i32, memref<3xi32>, index) -> ()
    %x_i32 = "memref.load"(%m_i32, %c) {"nontemporal" = false} : (memref<3xi32>, index) -> (i32)
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %v_f32, %v_i32, %r, %c, %m_f32, %m_i32 = "test.op"() : () -> (f32, i32, index, index, memref<3x2xf32>, memref<3xi32>)
// CHECK-NEXT:   %0 = builtin.unrealized_conversion_cast %v_f32 : f32 to !riscv.freg<>
// CHECK-NEXT:   %1 = builtin.unrealized_conversion_cast %m_f32 : memref<3x2xf32> to !riscv.reg<>
// CHECK-NEXT:   %2 = builtin.unrealized_conversion_cast %r : index to !riscv.reg<>
// CHECK-NEXT:   %3 = builtin.unrealized_conversion_cast %c : index to !riscv.reg<>
// CHECK-NEXT:   %4 = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:   %5 = riscv.mul %4, %2 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %6 = riscv.add %5, %3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %7 = riscv.li 4 : () -> !riscv.reg<>
// CHECK-NEXT:   %8 = riscv.mul %6, %7 {"comment" = "multiply by element size"} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %9 = riscv.add %1, %8 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   riscv.fsw %9, %0, 0 {"comment" = "store float value to memref of shape (3, 2)"} : (!riscv.reg<>, !riscv.freg<>) -> ()
// CHECK-NEXT:   %10 = builtin.unrealized_conversion_cast %m_f32 : memref<3x2xf32> to !riscv.reg<>
// CHECK-NEXT:   %11 = builtin.unrealized_conversion_cast %r : index to !riscv.reg<>
// CHECK-NEXT:   %12 = builtin.unrealized_conversion_cast %c : index to !riscv.reg<>
// CHECK-NEXT:   %13 = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:   %14 = riscv.mul %13, %11 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %15 = riscv.add %14, %12 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %16 = riscv.li 4 : () -> !riscv.reg<>
// CHECK-NEXT:   %17 = riscv.mul %15, %16 {"comment" = "multiply by element size"} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %18 = riscv.add %10, %17 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %x_f32 = riscv.flw %18, 0 {"comment" = "load value from memref of shape (3, 2)"} : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-NEXT:   %x_f32_1 = builtin.unrealized_conversion_cast %x_f32 : !riscv.freg<> to f32
// CHECK-NEXT:   %19 = builtin.unrealized_conversion_cast %v_i32 : i32 to !riscv.reg<>
// CHECK-NEXT:   %20 = builtin.unrealized_conversion_cast %m_i32 : memref<3xi32> to !riscv.reg<>
// CHECK-NEXT:   %21 = builtin.unrealized_conversion_cast %c : index to !riscv.reg<>
// CHECK-NEXT:   %22 = riscv.li 4 : () -> !riscv.reg<>
// CHECK-NEXT:   %23 = riscv.mul %21, %22 {"comment" = "multiply by element size"} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %24 = riscv.add %20, %23 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   riscv.sw %24, %19, 0 {"comment" = "store int value to memref of shape (3,)"} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:   %25 = builtin.unrealized_conversion_cast %m_i32 : memref<3xi32> to !riscv.reg<>
// CHECK-NEXT:   %26 = builtin.unrealized_conversion_cast %c : index to !riscv.reg<>
// CHECK-NEXT:   %27 = riscv.li 4 : () -> !riscv.reg<>
// CHECK-NEXT:   %28 = riscv.mul %26, %27 {"comment" = "multiply by element size"} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %29 = riscv.add %25, %28 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %x_i32 = riscv.lw %29, 0 {"comment" = "load value from memref of shape (3,)"} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %x_i32_1 = builtin.unrealized_conversion_cast %x_i32 : !riscv.reg<> to i32
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

// -----

builtin.module {
    %v, %d0, %d1, %d2, %m = "test.op"() : () -> (f32, index, index, index, memref<3x2x1xf32>)
    "memref.store"(%v, %m, %d0, %d1, %d2) {"nontemporal" = false} : (f32, memref<3x2x1xf32>, index, index, index) -> ()
}

// CHECK:      Unsupported memref shape (3, 2, 1), only support 1D and 2D memrefs.
