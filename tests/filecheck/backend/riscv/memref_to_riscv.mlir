// RUN: xdsl-opt -p convert-memref-to-riscv  --split-input-file --verify-diagnostics %s | filecheck %s

// CHECK:      builtin.module {

// CHECK-NEXT:   %v_f32, %v_f64, %v_i32 = "test.op"() : () -> (f32, f64, i32)
// CHECK-NEXT:   %r, %c = "test.op"() : () -> (index, index)
// CHECK-NEXT:   %m_f32, %m_f64, %m_i32, %m_scalar_i32 = "test.op"() : () -> (memref<3x2xf32>, memref<3x2xf64>, memref<3xi32>, memref<i32>)
%v_f32, %v_f64, %v_i32 = "test.op"() : () -> (f32, f64, i32)
%r, %c = "test.op"() : () -> (index, index)
%m_f32, %m_f64, %m_i32, %m_scalar_i32 = "test.op"() : () -> (memref<3x2xf32>, memref<3x2xf64>, memref<3xi32>, memref<i32>)

// CHECK-NEXT:   %0 = builtin.unrealized_conversion_cast %v_f32 : f32 to !riscv.freg
// CHECK-NEXT:   %1 = builtin.unrealized_conversion_cast %m_f32 : memref<3x2xf32> to !riscv.reg
// CHECK-NEXT:   %2 = builtin.unrealized_conversion_cast %r : index to !riscv.reg
// CHECK-NEXT:   %3 = builtin.unrealized_conversion_cast %c : index to !riscv.reg
// CHECK-NEXT:   %4 = riscv.li 2 : !riscv.reg
// CHECK-NEXT:   %5 = riscv.mul %2, %4 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %6 = riscv.add %5, %3 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %7 = riscv.li 4 : !riscv.reg
// CHECK-NEXT:   %8 = riscv.mul %6, %7 {"comment" = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %9 = riscv.add %1, %8 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   riscv.fsw %9, %0, 0 {"comment" = "store float value to memref of shape (3, 2)"} : (!riscv.reg, !riscv.freg) -> ()
memref.store %v_f32, %m_f32[%r, %c] {"nontemporal" = false} : memref<3x2xf32>

// CHECK-NEXT:   %10 = builtin.unrealized_conversion_cast %m_f32 : memref<3x2xf32> to !riscv.reg
// CHECK-NEXT:   %11 = builtin.unrealized_conversion_cast %r : index to !riscv.reg
// CHECK-NEXT:   %12 = builtin.unrealized_conversion_cast %c : index to !riscv.reg
// CHECK-NEXT:   %13 = riscv.li 2 : !riscv.reg
// CHECK-NEXT:   %14 = riscv.mul %11, %13 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %15 = riscv.add %14, %12 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %16 = riscv.li 4 : !riscv.reg
// CHECK-NEXT:   %17 = riscv.mul %15, %16 {"comment" = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %18 = riscv.add %10, %17 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %x_f32 = riscv.flw %18, 0 {"comment" = "load float from memref of shape (3, 2)"} : (!riscv.reg) -> !riscv.freg
// CHECK-NEXT:   %x_f32_1 = builtin.unrealized_conversion_cast %x_f32 : !riscv.freg to f32
%x_f32 = memref.load %m_f32[%r, %c] {"nontemporal" = false} : memref<3x2xf32>

// CHECK-NEXT:   %19 = builtin.unrealized_conversion_cast %v_i32 : i32 to !riscv.reg
// CHECK-NEXT:   %20 = builtin.unrealized_conversion_cast %m_i32 : memref<3xi32> to !riscv.reg
// CHECK-NEXT:   %21 = builtin.unrealized_conversion_cast %c : index to !riscv.reg
// CHECK-NEXT:   %22 = riscv.li 4 : !riscv.reg
// CHECK-NEXT:   %23 = riscv.mul %21, %22 {"comment" = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %24 = riscv.add %20, %23 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   riscv.sw %24, %19, 0 {"comment" = "store int value to memref of shape (3,)"} : (!riscv.reg, !riscv.reg) -> ()
memref.store %v_i32, %m_i32[%c] {"nontemporal" = false} : memref<3xi32>

// CHECK-NEXT:   %25 = builtin.unrealized_conversion_cast %m_i32 : memref<3xi32> to !riscv.reg
// CHECK-NEXT:   %26 = builtin.unrealized_conversion_cast %c : index to !riscv.reg
// CHECK-NEXT:   %27 = riscv.li 4 : !riscv.reg
// CHECK-NEXT:   %28 = riscv.mul %26, %27 {"comment" = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %29 = riscv.add %25, %28 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %x_i32 = riscv.lw %29, 0 {"comment" = "load word from memref of shape (3,)"} : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %x_i32_1 = builtin.unrealized_conversion_cast %x_i32 : !riscv.reg to i32
%x_i32 = memref.load %m_i32[%c] {"nontemporal" = false} : memref<3xi32>

// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %v_f64 : f64 to !riscv.freg
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %m_f64 : memref<3x2xf64> to !riscv.reg
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %r : index to !riscv.reg
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %c : index to !riscv.reg
// CHECK-NEXT:   %{{.*}} = riscv.li 2 : !riscv.reg
// CHECK-NEXT:   %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %{{.*}} = riscv.li 8 : !riscv.reg
// CHECK-NEXT:   %{{.*}} = riscv.mul %{{.*}}, %{{.*}} {"comment" = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   riscv.fsd %{{.*}}, %{{.*}}, 0 {"comment" = "store double value to memref of shape (3, 2)"} : (!riscv.reg, !riscv.freg) -> ()
memref.store %v_f64, %m_f64[%r, %c] {"nontemporal" = false} : memref<3x2xf64>

// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %m_scalar_i32 : memref<i32> to !riscv.reg
// CHECK-NEXT:   %scalar_x_i32 = riscv.lw %{{.*}}, 0 {"comment" = "load word from memref of shape ()"} : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %scalar_x_i32_1 = builtin.unrealized_conversion_cast %scalar_x_i32 : !riscv.reg to i32
%scalar_x_i32 = memref.load %m_scalar_i32[] {"nontemporal" = false} : memref<i32>

// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %scalar_x_i32_1 : i32 to !riscv.reg
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %m_scalar_i32 : memref<i32> to !riscv.reg
// CHECK-NEXT:   riscv.sw %{{.*}}, %{{.*}}, 0 {"comment" = "store int value to memref of shape ()"} : (!riscv.reg, !riscv.reg) -> ()
memref.store %scalar_x_i32, %m_scalar_i32[] {"nontemporal" = false} : memref<i32>

// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %m_f64 : memref<3x2xf64> to !riscv.reg
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %r : index to !riscv.reg
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %c : index to !riscv.reg
// CHECK-NEXT:   %{{.*}} = riscv.li 2 : !riscv.reg
// CHECK-NEXT:   %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %{{.*}} = riscv.li 8 : !riscv.reg
// CHECK-NEXT:   %{{.*}} = riscv.mul %{{.*}}, %{{.*}} {"comment" = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:   %x_f64 = riscv.fld %{{.*}}, 0 {"comment" = "load double from memref of shape (3, 2)"} : (!riscv.reg) -> !riscv.freg
// CHECK-NEXT:   %x_f64_1 = builtin.unrealized_conversion_cast %x_f64 : !riscv.freg to f64
%x_f64 = memref.load %m_f64[%r, %c] {"nontemporal" = false} : memref<3x2xf64>

// CHECK-NEXT:   riscv.assembly_section ".data" {
// CHECK-NEXT:       riscv.label "global"
// CHECK-NEXT:       riscv.directive ".word" "0x0,0x3ff00000,0x0,0x40000000"
// CHECK-NEXT:   }
"memref.global"() <{"sym_name" = "global", "sym_visibility" = "public", "type" = memref<2x3xf64>, "initial_value" = dense<[1, 2]> : tensor<2xi32>}> : () -> ()

// CHECK-NEXT:   %{{.*}} riscv.li "global" : !riscv.reg
// CHECK-NEXT:   %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.reg to memref<2xi32>
%global = memref.get_global @global : memref<2xi32>

// CHECK-NEXT: }

// -----

// CHECK:       builtin.module {

// CHECK-NEXT:    %m0 = riscv.li 4 {"comment" = "memref alloc size"} : !riscv.reg
// CHECK-NEXT:    %m0_1 = riscv.mv %m0 : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:    %m0_2 = riscv_func.call @malloc(%m0_1) : (!riscv.reg<a0>) -> !riscv.reg<a0>
// CHECK-NEXT:    %m0_3 = riscv.mv %m0_2 : (!riscv.reg<a0>) -> !riscv.reg
// CHECK-NEXT:    %m0_4 = builtin.unrealized_conversion_cast %m0_3 : !riscv.reg to memref<1x1xf32>
%m0 = memref.alloc() : memref<1x1xf32>

// CHECK-NEXT:    %m1 = riscv.li 8 {"comment" = "memref alloc size"} : !riscv.reg
// CHECK-NEXT:    %m1_1 = riscv.mv %m1 : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:    %m1_2 = riscv_func.call @malloc(%m1_1) : (!riscv.reg<a0>) -> !riscv.reg<a0>
// CHECK-NEXT:    %m1_3 = riscv.mv %m1_2 : (!riscv.reg<a0>) -> !riscv.reg
// CHECK-NEXT:    %m1_4 = builtin.unrealized_conversion_cast %m1_3 : !riscv.reg to memref<1x1xf64>
%m1 = memref.alloc() : memref<1x1xf64>

// Check that the malloc external function is declared after lowering

// CHECK-NEXT:    riscv_func.func private @malloc(!riscv.reg<a0>) -> !riscv.reg<a0>
// CHECK-NEXT:  }

// -----

// CHECK:       builtin.module {

// CHECK-NEXT:    %m = "test.op"() : () -> memref<1x1xf32>
%m = "test.op"() : () -> memref<1x1xf32>

// CHECK-NEXT:    %{{.*}} = builtin.unrealized_conversion_cast %m : memref<1x1xf32> to !riscv.reg<a0>
// CHECK-NEXT:    riscv_func.call @free(%{{.*}}) : (!riscv.reg<a0>) -> ()
"memref.dealloc"(%m) : (memref<1x1xf32>) -> ()

// Check that the dealloc external function is declared after lowering

// CHECK-NEXT:    riscv_func.func private @free(!riscv.reg<a0>) -> ()
// CHECK-NEXT:  }

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    %v, %d0, %m = "test.op"() : () -> (i8, index, memref<1xi8>)
%v, %d0, %m = "test.op"() : () -> (i8, index, memref<1xi8>)

// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %v : i8 to !riscv.reg
// CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %m : memref<1xi8> to !riscv.reg
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %d0 : index to !riscv.reg
// CHECK-NEXT:    %3 = riscv.li 1 : !riscv.reg
// CHECK-NEXT:    %4 = riscv.mul %2, %3 {"comment" = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %5 = riscv.add %1, %4 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    riscv.sw %5, %0, 0 {"comment" = "store int value to memref of shape (1,)"} : (!riscv.reg, !riscv.reg) -> ()
memref.store %v, %m[%d0] {"nontemporal" = false} : memref<1xi8>

// CHECK-NEXT:  }

// -----

// CHECK:       builtin.module {

// CHECK-NEXT:    %v, %d0, %m = "test.op"() : () -> (i16, index, memref<1xi16>)
%v, %d0, %m = "test.op"() : () -> (i16, index, memref<1xi16>)

// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %v : i16 to !riscv.reg
// CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %m : memref<1xi16> to !riscv.reg
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %d0 : index to !riscv.reg
// CHECK-NEXT:    %3 = riscv.li 2 : !riscv.reg
// CHECK-NEXT:    %4 = riscv.mul %2, %3 {"comment" = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %5 = riscv.add %1, %4 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    riscv.sw %5, %0, 0 {"comment" = "store int value to memref of shape (1,)"} : (!riscv.reg, !riscv.reg) -> ()
memref.store %v, %m[%d0] {"nontemporal" = false} : memref<1xi16>

// CHECK-NEXT:  }

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    %v, %d0, %m = "test.op"() : () -> (i64, index, memref<1xi64>)

%v, %d0, %m = "test.op"() : () -> (i64, index, memref<1xi64>)

// CHECK-NEXT:    %0 = builtin.unrealized_conversion_cast %v : i64 to !riscv.reg
// CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %m : memref<1xi64> to !riscv.reg
// CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %d0 : index to !riscv.reg
// CHECK-NEXT:    %3 = riscv.li 8 : !riscv.reg
// CHECK-NEXT:    %4 = riscv.mul %2, %3 {"comment" = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %5 = riscv.add %1, %4 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    riscv.sw %5, %0, 0 {"comment" = "store int value to memref of shape (1,)"} : (!riscv.reg, !riscv.reg) -> ()
memref.store %v, %m[%d0] {"nontemporal" = false} : memref<1xi64>

// CHECK-NEXT:  }

// -----

%m = "test.op"() : () -> memref<2x3xf64, strided<[6, 1], offset: ?>>
%i0, %i1 = "test.op"() : () -> (index, index)

// CHECK: %0 = builtin.unrealized_conversion_cast %m : memref<2x3xf64, strided<[6, 1], offset: ?>> to !riscv.reg
// CHECK-NEXT: %1 = builtin.unrealized_conversion_cast %i0 : index to !riscv.reg
// CHECK-NEXT: %2 = builtin.unrealized_conversion_cast %i1 : index to !riscv.reg
// CHECK-NEXT: %3 = riscv.li 6
// CHECK-NEXT: %4 = riscv.mul %1, %3
// CHECK-NEXT: %5 = riscv.add %4, %2
// CHECK-NEXT: %6 = riscv.li 8
// CHECK-NEXT: %7 = riscv.mul %5, %6
// CHECK-NEXT: %8 = riscv.add %0, %7
// CHECK-NEXT: %v = riscv.fld %8, 0
// CHECK-NEXT: %v_1 = builtin.unrealized_conversion_cast %v : !riscv.freg to f64
%v = memref.load %m[%i0, %i1] : memref<2x3xf64, strided<[6, 1], offset: ?>>

// -----

%m = "test.op"() : () -> memref<2x3xf64, strided<[6, 1], offset: ?>>
%v = "test.op"() : () -> f64
%i0, %i1 = "test.op"() : () -> (index, index)

memref.store %v, %m[%i0, %i1] : memref<2x3xf64, strided<[6, 1], offset: ?>>

// CHECK: %0 = builtin.unrealized_conversion_cast %v : f64 to !riscv.freg
// CHECK-NEXT: %1 = builtin.unrealized_conversion_cast %m : memref<2x3xf64, strided<[6, 1], offset: ?>> to !riscv.reg
// CHECK-NEXT: %2 = builtin.unrealized_conversion_cast %i0 : index to !riscv.reg
// CHECK-NEXT: %3 = builtin.unrealized_conversion_cast %i1 : index to !riscv.reg
// CHECK-NEXT: %4 = riscv.li 6
// CHECK-NEXT: %5 = riscv.mul %2, %4
// CHECK-NEXT: %6 = riscv.add %5, %3
// CHECK-NEXT: %7 = riscv.li 8
// CHECK-NEXT: %8 = riscv.mul %6, %7
// CHECK-NEXT: %9 = riscv.add %1, %8
// CHECK-NEXT: riscv.fsd %9, %0, 0

// -----

%m = "test.op"() : () -> memref<2xf64, strided<[?]>>
%i0 = "test.op"() : () -> index
%v = memref.load %m[%i0] : memref<2xf64, strided<[?]>>

// CHECK: MemRef memref<2xf64, strided<[?]>> with dynamic stride is not yet implemented
