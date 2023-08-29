// RUN: xdsl-opt -p convert-memref-to-riscv  --split-input-file --verify-diagnostics %s | filecheck %s

builtin.module {
    %v_f32, %v_i32, %r, %c, %m_f32, %m_i32 = "test.op"() : () -> (f32, i32, index, index, memref<3x2xf32>, memref<3x2xi32>)
    "memref.store"(%v_f32, %m_f32, %r, %c) {"nontemporal" = false} : (f32, memref<3x2xf32>, index, index) -> ()
    %x_f32 = "memref.load"(%m_f32, %r, %c) {"nontemporal" = false} : (memref<3x2xf32>, index, index) -> (f32)
    "memref.store"(%v_i32, %m_i32, %r, %c) {"nontemporal" = false} : (i32, memref<3x2xi32>, index, index) -> ()
    %x_i32 = "memref.load"(%m_i32, %r, %c) {"nontemporal" = false} : (memref<3x2xi32>, index, index) -> (i32)
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %v_f32, %v_i32, %r, %c, %m_f32, %m_i32 = "test.op"() : () -> (f32, i32, index, index, memref<3x2xf32>, memref<3x2xi32>)
// CHECK-NEXT:   %0 = builtin.unrealized_conversion_cast %v_f32 : f32 to !riscv.freg<>
// CHECK-NEXT:   %1 = builtin.unrealized_conversion_cast %m_f32 : memref<3x2xf32> to !riscv.reg<>
// CHECK-NEXT:   %2 = builtin.unrealized_conversion_cast %r : index to !riscv.reg<>
// CHECK-NEXT:   %3 = builtin.unrealized_conversion_cast %c : index to !riscv.reg<>
// CHECK-NEXT:   %4 = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:   %5 = riscv.mul %4, %2 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %6 = riscv.add %5, %3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %7 = riscv.slli %6, 2 {"comment" = "mutiply by elm size"} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %8 = riscv.add %1, %7 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   riscv.fsw %8, %0, 0 {"comment" = "store float value to memref of shape (3, 2)"} : (!riscv.reg<>, !riscv.freg<>) -> ()
// CHECK-NEXT:   %9 = builtin.unrealized_conversion_cast %m_f32 : memref<3x2xf32> to !riscv.reg<>
// CHECK-NEXT:   %10 = builtin.unrealized_conversion_cast %r : index to !riscv.reg<>
// CHECK-NEXT:   %11 = builtin.unrealized_conversion_cast %c : index to !riscv.reg<>
// CHECK-NEXT:   %12 = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:   %13 = riscv.mul %12, %10 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %14 = riscv.add %13, %11 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %15 = riscv.slli %14, 2 {"comment" = "mutiply by elm size"} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %16 = riscv.add %9, %15 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %x_f32 = riscv.flw %16, 0 {"comment" = "load value from memref of shape (3, 2)"} : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-NEXT:   %x_f32_1 = builtin.unrealized_conversion_cast %x_f32 : !riscv.freg<> to f32
// CHECK-NEXT:   %17 = builtin.unrealized_conversion_cast %v_i32 : i32 to !riscv.reg<>
// CHECK-NEXT:   %18 = builtin.unrealized_conversion_cast %m_i32 : memref<3x2xi32> to !riscv.reg<>
// CHECK-NEXT:   %19 = builtin.unrealized_conversion_cast %r : index to !riscv.reg<>
// CHECK-NEXT:   %20 = builtin.unrealized_conversion_cast %c : index to !riscv.reg<>
// CHECK-NEXT:   %21 = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:   %22 = riscv.mul %21, %19 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %23 = riscv.add %22, %20 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %24 = riscv.slli %23, 2 {"comment" = "mutiply by elm size"} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %25 = riscv.add %18, %24 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   riscv.sw %25, %17, 0 {"comment" = "store int value to memref of shape (3, 2)"} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:   %26 = builtin.unrealized_conversion_cast %m_i32 : memref<3x2xi32> to !riscv.reg<>
// CHECK-NEXT:   %27 = builtin.unrealized_conversion_cast %r : index to !riscv.reg<>
// CHECK-NEXT:   %28 = builtin.unrealized_conversion_cast %c : index to !riscv.reg<>
// CHECK-NEXT:   %29 = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:   %30 = riscv.mul %29, %27 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %31 = riscv.add %30, %28 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %32 = riscv.slli %31, 2 {"comment" = "mutiply by elm size"} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %33 = riscv.add %26, %32 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %x_i32 = riscv.lw %33, 0 {"comment" = "load value from memref of shape (3, 2)"} : (!riscv.reg<>) -> !riscv.reg<>
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
