// RUN: xdsl-opt -p convert-memref-to-riscv  --split-input-file --verify-diagnostics %s | filecheck %s

builtin.module {
    %v_f32, %v_f64, %v_i32, %r, %c, %m_f32, %m_f64, %m_i32, %m_scalar_i32 = "test.op"() : () -> (f32, f64, i32, index, index, memref<3x2xf32>, memref<3x2xf64>, memref<3xi32>, memref<i32>)
    "memref.store"(%v_f32, %m_f32, %r, %c) {"nontemporal" = false} : (f32, memref<3x2xf32>, index, index) -> ()
    %x_f32 = "memref.load"(%m_f32, %r, %c) {"nontemporal" = false} : (memref<3x2xf32>, index, index) -> (f32)
    "memref.store"(%v_i32, %m_i32, %c) {"nontemporal" = false} : (i32, memref<3xi32>, index) -> ()
    %x_i32 = "memref.load"(%m_i32, %c) {"nontemporal" = false} : (memref<3xi32>, index) -> (i32)
    "memref.store"(%v_f64, %m_f64, %r, %c) {"nontemporal" = false} : (f64, memref<3x2xf64>, index, index) -> ()
    %scalar_x_i32 = "memref.load"(%m_scalar_i32) {"nontemporal" = false} : (memref<i32>) -> (i32)
    "memref.store"(%scalar_x_i32, %m_scalar_i32) {"nontemporal" = false} : (i32, memref<i32>) -> ()
    %x_f64 = "memref.load"(%m_f64, %r, %c) {"nontemporal" = false} : (memref<3x2xf64>, index, index) -> (f64)
    "memref.global"() {"sym_name" = "global", "type" = memref<2x3xf64>, "initial_value" = dense<[1, 2]> : tensor<2xi32>, "sym_visibility" = "public"} : () -> ()
    %global = "memref.get_global"() {"name" = @global} : () -> memref<2xi32>
}

// CHECK:      builtin.module {
// CHECK-NEXT:     %v_f32, %v_f64, %v_i32, %r, %c, %m_f32, %m_f64, %m_i32, %m_scalar_i32 = "test.op"() : () -> (f32, f64, i32, index, index, memref<3x2xf32>, memref<3x2xf64>, memref<3xi32>, memref<i32>)
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
// CHECK-NEXT:   %x_f32 = riscv.flw %18, 0 {"comment" = "load float from memref of shape (3, 2)"} : (!riscv.reg<>) -> !riscv.freg<>
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
// CHECK-NEXT:   %x_i32 = riscv.lw %29, 0 {"comment" = "load word from memref of shape (3,)"} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %x_i32_1 = builtin.unrealized_conversion_cast %x_i32 : !riscv.reg<> to i32
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %v_f64 : f64 to !riscv.freg<>
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %m_f64 : memref<3x2xf64> to !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %r : index to !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %c : index to !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = riscv.mul %{{.*}}, %{{.*}} {"comment" = "multiply by element size"} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   riscv.fsd %{{.*}}, %{{.*}}, 0 {"comment" = "store double value to memref of shape (3, 2)"} : (!riscv.reg<>, !riscv.freg<>) -> ()
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %m_scalar_i32 : memref<i32> to !riscv.reg<>
// CHECK-NEXT:   %scalar_x_i32 = riscv.lw %{{.*}}, 0 {"comment" = "load word from memref of shape ()"} : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %scalar_x_i32_1 = builtin.unrealized_conversion_cast %scalar_x_i32 : !riscv.reg<> to i32
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %scalar_x_i32_1 : i32 to !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %m_scalar_i32 : memref<i32> to !riscv.reg<>
// CHECK-NEXT:   riscv.sw %{{.*}}, %{{.*}}, 0 {"comment" = "store int value to memref of shape ()"} : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %m_f64 : memref<3x2xf64> to !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %r : index to !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %c : index to !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = riscv.mul %{{.*}}, %{{.*}} {"comment" = "multiply by element size"} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:   %x_f64 = riscv.fld %{{.*}}, 0 {"comment" = "load double from memref of shape (3, 2)"} : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-NEXT:   %x_f64_1 = builtin.unrealized_conversion_cast %x_f64 : !riscv.freg<> to f64
// CHECK-NEXT:   riscv.assembly_section ".data" {
// CHECK-NEXT:       riscv.label "global"
// CHECK-NEXT:       riscv.directive ".word" "0x0,0x3ff00000,0x0,0x40000000"
// CHECK-NEXT:   }
// CHECK-NEXT:   %{{.*}} riscv.li "global" : () -> !riscv.reg<>
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.reg<> to memref<2xi32>
// CHECK-NEXT: }

// -----

builtin.module {
    %m = "memref.alloc"() {"operandSegmentSizes" = array<i32: 0, 0>} : () -> memref<1x1xf32>
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
    %v, %d0, %m = "test.op"() : () -> (i8, index, memref<1xi8>)
    "memref.store"(%v, %m, %d0) {"nontemporal" = false} : (i8, memref<1xi8>, index) -> ()
}

// CHECK:      Unsupported memref element type for riscv lowering: i8

// -----

builtin.module {
    %v, %d0, %m = "test.op"() : () -> (i16, index, memref<1xi16>)
    "memref.store"(%v, %m, %d0) {"nontemporal" = false} : (i16, memref<1xi16>, index) -> ()
}

// CHECK:      Unsupported memref element type for riscv lowering: i16

// -----

builtin.module {
    %v, %d0, %m = "test.op"() : () -> (i64, index, memref<1xi64>)
    "memref.store"(%v, %m, %d0) {"nontemporal" = false} : (i64, memref<1xi64>, index) -> ()
}

// CHECK:      Unsupported memref element type for riscv lowering: i64
