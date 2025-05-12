// RUN: xdsl-opt -p convert-memref-to-riscv  --split-input-file --verify-diagnostics %s | filecheck %s

// CHECK:      builtin.module {

// CHECK-NEXT:   %v_f32, %v_f64, %v_i32 = "test.op"() : () -> (f32, f64, i32)
// CHECK-NEXT:   %r, %c = "test.op"() : () -> (index, index)
// CHECK-NEXT:   %m_f32, %m_f64, %m_i32, %m_scalar_i32 = "test.op"() : () -> (memref<3x2xf32>, memref<3x2xf64>, memref<3xi32>, memref<i32>)
%v_f32, %v_f64, %v_i32 = "test.op"() : () -> (f32, f64, i32)
%r, %c = "test.op"() : () -> (index, index)
%m_f32, %m_f64, %m_i32, %m_scalar_i32 = "test.op"() : () -> (memref<3x2xf32>, memref<3x2xf64>, memref<3xi32>, memref<i32>)

// CHECK-NEXT:    %v_f32_1 = builtin.unrealized_conversion_cast %v_f32 : f32 to !riscv.freg
// CHECK-NEXT:    %m_f32_1 = builtin.unrealized_conversion_cast %m_f32 : memref<3x2xf32> to !riscv.reg
// CHECK-NEXT:    %r_1 = builtin.unrealized_conversion_cast %r : index to !riscv.reg
// CHECK-NEXT:    %c_1 = builtin.unrealized_conversion_cast %c : index to !riscv.reg
// CHECK-NEXT:    %pointer_dim_stride = riscv.li 2 : !riscv.reg
// CHECK-NEXT:    %pointer_dim_offset = riscv.mul %r_1, %pointer_dim_stride : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %pointer_offset = riscv.add %pointer_dim_offset, %c_1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %bytes_per_element = riscv.li 4 : !riscv.reg
// CHECK-NEXT:    %scaled_pointer_offset = riscv.mul %pointer_offset, %bytes_per_element {comment = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %offset_pointer = riscv.add %m_f32_1, %scaled_pointer_offset : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    riscv.fsw %offset_pointer, %v_f32_1, 0 {comment = "store float value to memref of shape (3, 2)"} : (!riscv.reg, !riscv.freg) -> ()
memref.store %v_f32, %m_f32[%r, %c] {"nontemporal" = false} : memref<3x2xf32>

// CHECK-NEXT:    %m_f32_2 = builtin.unrealized_conversion_cast %m_f32 : memref<3x2xf32> to !riscv.reg
// CHECK-NEXT:    %r_2 = builtin.unrealized_conversion_cast %r : index to !riscv.reg
// CHECK-NEXT:    %c_2 = builtin.unrealized_conversion_cast %c : index to !riscv.reg
// CHECK-NEXT:    %pointer_dim_stride_1 = riscv.li 2 : !riscv.reg
// CHECK-NEXT:    %pointer_dim_offset_1 = riscv.mul %r_2, %pointer_dim_stride_1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %pointer_offset_1 = riscv.add %pointer_dim_offset_1, %c_2 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %bytes_per_element_1 = riscv.li 4 : !riscv.reg
// CHECK-NEXT:    %scaled_pointer_offset_1 = riscv.mul %pointer_offset_1, %bytes_per_element_1 {comment = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %offset_pointer_1 = riscv.add %m_f32_2, %scaled_pointer_offset_1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %x_f32 = riscv.flw %offset_pointer_1, 0 {comment = "load float from memref of shape (3, 2)"} : (!riscv.reg) -> !riscv.freg
// CHECK-NEXT:    %x_f32_1 = builtin.unrealized_conversion_cast %x_f32 : !riscv.freg to f32
%x_f32 = memref.load %m_f32[%r, %c] {"nontemporal" = false} : memref<3x2xf32>

// CHECK-NEXT:    %v_i32_1 = builtin.unrealized_conversion_cast %v_i32 : i32 to !riscv.reg
// CHECK-NEXT:    %m_i32_1 = builtin.unrealized_conversion_cast %m_i32 : memref<3xi32> to !riscv.reg
// CHECK-NEXT:    %c_3 = builtin.unrealized_conversion_cast %c : index to !riscv.reg
// CHECK-NEXT:    %bytes_per_element_2 = riscv.li 4 : !riscv.reg
// CHECK-NEXT:    %scaled_pointer_offset_2 = riscv.mul %c_3, %bytes_per_element_2 {comment = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %offset_pointer_2 = riscv.add %m_i32_1, %scaled_pointer_offset_2 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    riscv.sw %offset_pointer_2, %v_i32_1, 0 {comment = "store int value to memref of shape (3,)"} : (!riscv.reg, !riscv.reg) -> ()
memref.store %v_i32, %m_i32[%c] {"nontemporal" = false} : memref<3xi32>

// CHECK-NEXT:    %m_i32_2 = builtin.unrealized_conversion_cast %m_i32 : memref<3xi32> to !riscv.reg
// CHECK-NEXT:    %c_4 = builtin.unrealized_conversion_cast %c : index to !riscv.reg
// CHECK-NEXT:    %bytes_per_element_3 = riscv.li 4 : !riscv.reg
// CHECK-NEXT:    %scaled_pointer_offset_3 = riscv.mul %c_4, %bytes_per_element_3 {comment = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %offset_pointer_3 = riscv.add %m_i32_2, %scaled_pointer_offset_3 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %x_i32 = riscv.lw %offset_pointer_3, 0 {comment = "load word from memref of shape (3,)"} : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %x_i32_1 = builtin.unrealized_conversion_cast %x_i32 : !riscv.reg to i32
%x_i32 = memref.load %m_i32[%c] {"nontemporal" = false} : memref<3xi32>

// CHECK-NEXT:    %v_f64_1 = builtin.unrealized_conversion_cast %v_f64 : f64 to !riscv.freg
// CHECK-NEXT:    %m_f64_1 = builtin.unrealized_conversion_cast %m_f64 : memref<3x2xf64> to !riscv.reg
// CHECK-NEXT:    %r_3 = builtin.unrealized_conversion_cast %r : index to !riscv.reg
// CHECK-NEXT:    %c_5 = builtin.unrealized_conversion_cast %c : index to !riscv.reg
// CHECK-NEXT:    %pointer_dim_stride_2 = riscv.li 2 : !riscv.reg
// CHECK-NEXT:    %pointer_dim_offset_2 = riscv.mul %r_3, %pointer_dim_stride_2 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %pointer_offset_2 = riscv.add %pointer_dim_offset_2, %c_5 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %bytes_per_element_4 = riscv.li 8 : !riscv.reg
// CHECK-NEXT:    %scaled_pointer_offset_4 = riscv.mul %pointer_offset_2, %bytes_per_element_4 {comment = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %offset_pointer_4 = riscv.add %m_f64_1, %scaled_pointer_offset_4 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    riscv.fsd %offset_pointer_4, %v_f64_1, 0 {comment = "store double value to memref of shape (3, 2)"} : (!riscv.reg, !riscv.freg) -> ()
memref.store %v_f64, %m_f64[%r, %c] {"nontemporal" = false} : memref<3x2xf64>

// CHECK-NEXT:    %m_scalar_i32_1 = builtin.unrealized_conversion_cast %m_scalar_i32 : memref<i32> to !riscv.reg
// CHECK-NEXT:    %scalar_x_i32 = riscv.lw %m_scalar_i32_1, 0 {comment = "load word from memref of shape ()"} : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %scalar_x_i32_1 = builtin.unrealized_conversion_cast %scalar_x_i32 : !riscv.reg to i32
%scalar_x_i32 = memref.load %m_scalar_i32[] {"nontemporal" = false} : memref<i32>

// CHECK-NEXT:    %scalar_x_i32_2 = builtin.unrealized_conversion_cast %scalar_x_i32_1 : i32 to !riscv.reg
// CHECK-NEXT:    %m_scalar_i32_2 = builtin.unrealized_conversion_cast %m_scalar_i32 : memref<i32> to !riscv.reg
// CHECK-NEXT:    riscv.sw %m_scalar_i32_2, %scalar_x_i32_2, 0 {comment = "store int value to memref of shape ()"} : (!riscv.reg, !riscv.reg) -> ()
memref.store %scalar_x_i32, %m_scalar_i32[] {"nontemporal" = false} : memref<i32>

// CHECK-NEXT:    %m_f64_2 = builtin.unrealized_conversion_cast %m_f64 : memref<3x2xf64> to !riscv.reg
// CHECK-NEXT:    %r_4 = builtin.unrealized_conversion_cast %r : index to !riscv.reg
// CHECK-NEXT:    %c_6 = builtin.unrealized_conversion_cast %c : index to !riscv.reg
// CHECK-NEXT:    %pointer_dim_stride_3 = riscv.li 2 : !riscv.reg
// CHECK-NEXT:    %pointer_dim_offset_3 = riscv.mul %r_4, %pointer_dim_stride_3 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %pointer_offset_3 = riscv.add %pointer_dim_offset_3, %c_6 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %bytes_per_element_5 = riscv.li 8 : !riscv.reg
// CHECK-NEXT:    %scaled_pointer_offset_5 = riscv.mul %pointer_offset_3, %bytes_per_element_5 {comment = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %offset_pointer_5 = riscv.add %m_f64_2, %scaled_pointer_offset_5 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %x_f64 = riscv.fld %offset_pointer_5, 0 {comment = "load double from memref of shape (3, 2)"} : (!riscv.reg) -> !riscv.freg
// CHECK-NEXT:    %x_f64_1 = builtin.unrealized_conversion_cast %x_f64 : !riscv.freg to f64
%x_f64 = memref.load %m_f64[%r, %c] {"nontemporal" = false} : memref<3x2xf64>

// CHECK-NEXT:   riscv.assembly_section ".data" {
// CHECK-NEXT:       riscv.label "global"
// CHECK-NEXT:       riscv.directive ".word" "0x0,0x3ff00000,0x0,0x40000000"
// CHECK-NEXT:   }
"memref.global"() <{"sym_name" = "global", "sym_visibility" = "public", "type" = memref<2xf64>, "initial_value" = dense<[1, 2]> : tensor<2xf64>}> : () -> ()

// CHECK-NEXT:    %global = riscv.li "global" : !riscv.reg
// CHECK-NEXT:    %global_1 = builtin.unrealized_conversion_cast %global : !riscv.reg to memref<2xi32>
%global = memref.get_global @global : memref<2xi32>

// CHECK-NEXT: }

// -----

// CHECK:       builtin.module {

// CHECK-NEXT:    %m0 = riscv.li 4 {comment = "memref alloc size"} : !riscv.reg
// CHECK-NEXT:    %m0_1 = riscv.mv %m0 : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:    %m0_2 = riscv_func.call @malloc(%m0_1) : (!riscv.reg<a0>) -> !riscv.reg<a0>
// CHECK-NEXT:    %m0_3 = riscv.mv %m0_2 : (!riscv.reg<a0>) -> !riscv.reg
// CHECK-NEXT:    %m0_4 = builtin.unrealized_conversion_cast %m0_3 : !riscv.reg to memref<1x1xf32>
%m0 = memref.alloc() : memref<1x1xf32>

// CHECK-NEXT:    %m1 = riscv.li 8 {comment = "memref alloc size"} : !riscv.reg
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

// CHECK-NEXT:    %{{.*}} = builtin.unrealized_conversion_cast %m : memref<1x1xf32> to !riscv.reg
// CHECK-NEXT:    %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:    riscv_func.call @free(%{{.*}}) : (!riscv.reg<a0>) -> ()
"memref.dealloc"(%m) : (memref<1x1xf32>) -> ()

// Check that the dealloc external function is declared after lowering

// CHECK-NEXT:    riscv_func.func private @free(!riscv.reg<a0>) -> ()
// CHECK-NEXT:  }

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    %v, %d0, %m = "test.op"() : () -> (i8, index, memref<1xi8>)
%v, %d0, %m = "test.op"() : () -> (i8, index, memref<1xi8>)

// CHECK-NEXT:    %v_1 = builtin.unrealized_conversion_cast %v : i8 to !riscv.reg
// CHECK-NEXT:    %m_1 = builtin.unrealized_conversion_cast %m : memref<1xi8> to !riscv.reg
// CHECK-NEXT:    %d0_1 = builtin.unrealized_conversion_cast %d0 : index to !riscv.reg
// CHECK-NEXT:    %bytes_per_element = riscv.li 1 : !riscv.reg
// CHECK-NEXT:    %scaled_pointer_offset = riscv.mul %d0_1, %bytes_per_element {comment = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %offset_pointer = riscv.add %m_1, %scaled_pointer_offset : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    riscv.sw %offset_pointer, %v_1, 0 {comment = "store int value to memref of shape (1,)"} : (!riscv.reg, !riscv.reg) -> ()
memref.store %v, %m[%d0] {"nontemporal" = false} : memref<1xi8>

// CHECK-NEXT:  }

// -----

// CHECK:       builtin.module {

// CHECK-NEXT:    %v, %d0, %m = "test.op"() : () -> (i16, index, memref<1xi16>)
%v, %d0, %m = "test.op"() : () -> (i16, index, memref<1xi16>)

// CHECK-NEXT:    %v_1 = builtin.unrealized_conversion_cast %v : i16 to !riscv.reg
// CHECK-NEXT:    %m_1 = builtin.unrealized_conversion_cast %m : memref<1xi16> to !riscv.reg
// CHECK-NEXT:    %d0_1 = builtin.unrealized_conversion_cast %d0 : index to !riscv.reg
// CHECK-NEXT:    %bytes_per_element = riscv.li 2 : !riscv.reg
// CHECK-NEXT:    %scaled_pointer_offset = riscv.mul %d0_1, %bytes_per_element {comment = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %offset_pointer = riscv.add %m_1, %scaled_pointer_offset : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    riscv.sw %offset_pointer, %v_1, 0 {comment = "store int value to memref of shape (1,)"} : (!riscv.reg, !riscv.reg) -> ()
memref.store %v, %m[%d0] {"nontemporal" = false} : memref<1xi16>

// CHECK-NEXT:  }

// -----

// CHECK:       builtin.module {
// CHECK-NEXT:    %v, %d0, %m = "test.op"() : () -> (i64, index, memref<1xi64>)

%v, %d0, %m = "test.op"() : () -> (i64, index, memref<1xi64>)

// CHECK-NEXT:    %v_1 = builtin.unrealized_conversion_cast %v : i64 to !riscv.reg
// CHECK-NEXT:    %m_1 = builtin.unrealized_conversion_cast %m : memref<1xi64> to !riscv.reg
// CHECK-NEXT:    %d0_1 = builtin.unrealized_conversion_cast %d0 : index to !riscv.reg
// CHECK-NEXT:    %bytes_per_element = riscv.li 8 : !riscv.reg
// CHECK-NEXT:    %scaled_pointer_offset = riscv.mul %d0_1, %bytes_per_element {comment = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %offset_pointer = riscv.add %m_1, %scaled_pointer_offset : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    riscv.sw %offset_pointer, %v_1, 0 {comment = "store int value to memref of shape (1,)"} : (!riscv.reg, !riscv.reg) -> ()
memref.store %v, %m[%d0] {"nontemporal" = false} : memref<1xi64>

// CHECK-NEXT:  }

// -----

// CHECK:       builtin.module {

// CHECK-NEXT:    %v_f64 = "test.op"() : () -> f64
// CHECK-NEXT:    %i0, %i1, %offset = "test.op"() : () -> (index, index, index)
// CHECK-NEXT:    %original = "test.op"() : () -> memref<4x3x2xf64>
%v_f64 = "test.op"() : () -> f64
%i0, %i1, %offset = "test.op"() : () -> (index, index, index)
%original = "test.op"() : () -> memref<4x3x2xf64>

// CHECK-NEXT:    %zero_subview = builtin.unrealized_conversion_cast %original : memref<4x3x2xf64> to memref<3x2xf64>
%zero_subview = memref.subview %original[0, 0, 0][1, 3, 2][1, 1, 1] : memref<4x3x2xf64> to memref<3x2xf64>

// CHECK-NEXT:    %static_subview = builtin.unrealized_conversion_cast %original : memref<4x3x2xf64> to !riscv.reg
// CHECK-NEXT:    %static_subview_1 = riscv.addi %static_subview, 48 {comment = "subview offset"} : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %static_subview_2 = builtin.unrealized_conversion_cast %static_subview_1 : !riscv.reg to memref<3x2xf64, strided<[2, 1], offset: 6>>
%static_subview = memref.subview %original[1, 0, 0][1, 3, 2][1, 1, 1] :
  memref<4x3x2xf64> to memref<3x2xf64, strided<[2, 1], offset: 6>>

// CHECK-NEXT:    %dynamic_subview = builtin.unrealized_conversion_cast %original : memref<4x3x2xf64> to !riscv.reg
// CHECK-NEXT:    %subview_dim_index = builtin.unrealized_conversion_cast %offset : index to !riscv.reg
// CHECK-NEXT:    %subview_dim_index_1 = riscv.li 0 : !riscv.reg
// CHECK-NEXT:    %subview_dim_index_2 = riscv.li 0 : !riscv.reg
// CHECK-NEXT:    %pointer_dim_stride = riscv.li 6 : !riscv.reg
// CHECK-NEXT:    %pointer_dim_offset = riscv.mul %subview_dim_index, %pointer_dim_stride : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %pointer_dim_stride_1 = riscv.li 2 : !riscv.reg
// CHECK-NEXT:    %pointer_dim_offset_1 = riscv.mul %subview_dim_index_1, %pointer_dim_stride_1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %pointer_offset = riscv.add %pointer_dim_offset, %pointer_dim_offset_1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %pointer_offset_1 = riscv.add %pointer_offset, %subview_dim_index_2 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %bytes_per_element = riscv.li 8 : !riscv.reg
// CHECK-NEXT:    %scaled_pointer_offset = riscv.mul %pointer_offset_1, %bytes_per_element {comment = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %offset_pointer = riscv.add %dynamic_subview, %scaled_pointer_offset : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %dynamic_subview_1 = builtin.unrealized_conversion_cast %offset_pointer : !riscv.reg to memref<3x2xf64, strided<[2, 1], offset: ?>>
%dynamic_subview = memref.subview %original[%offset, 0, 0][1, 3, 2][1, 1, 1] :
  memref<4x3x2xf64> to memref<3x2xf64, strided<[2, 1], offset: ?>>

// CHECK-NEXT:    %larger_original = "test.op"() : () -> memref<5x4x3x2xf64>
%larger_original = "test.op"() : () -> memref<5x4x3x2xf64>
// CHECK-NEXT:    %larger_dynamic_subview = builtin.unrealized_conversion_cast %larger_original : memref<5x4x3x2xf64> to !riscv.reg
// CHECK-NEXT:    %subview_dim_index_3 = builtin.unrealized_conversion_cast %offset : index to !riscv.reg
// CHECK-NEXT:    %subview_dim_index_4 = builtin.unrealized_conversion_cast %offset : index to !riscv.reg
// CHECK-NEXT:    %subview_dim_index_5 = riscv.li 0 : !riscv.reg
// CHECK-NEXT:    %subview_dim_index_6 = riscv.li 0 : !riscv.reg
// CHECK-NEXT:    %pointer_dim_stride_2 = riscv.li 24 : !riscv.reg
// CHECK-NEXT:    %pointer_dim_offset_2 = riscv.mul %subview_dim_index_3, %pointer_dim_stride_2 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %pointer_dim_stride_3 = riscv.li 6 : !riscv.reg
// CHECK-NEXT:    %pointer_dim_offset_3 = riscv.mul %subview_dim_index_4, %pointer_dim_stride_3 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %pointer_offset_2 = riscv.add %pointer_dim_offset_2, %pointer_dim_offset_3 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %pointer_dim_stride_4 = riscv.li 2 : !riscv.reg
// CHECK-NEXT:    %pointer_dim_offset_4 = riscv.mul %subview_dim_index_5, %pointer_dim_stride_4 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %pointer_offset_3 = riscv.add %pointer_offset_2, %pointer_dim_offset_4 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %pointer_offset_4 = riscv.add %pointer_offset_3, %subview_dim_index_6 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %bytes_per_element_1 = riscv.li 8 : !riscv.reg
// CHECK-NEXT:    %scaled_pointer_offset_1 = riscv.mul %pointer_offset_4, %bytes_per_element_1 {comment = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %offset_pointer_1 = riscv.add %larger_dynamic_subview, %scaled_pointer_offset_1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:    %larger_dynamic_subview_1 = builtin.unrealized_conversion_cast %offset_pointer_1 : !riscv.reg to memref<3x2xf64, strided<[2, 1], offset: ?>>
%larger_dynamic_subview = memref.subview %larger_original[%offset, %offset, 0, 0][1, 1, 3, 2][1, 1, 1, 1] :
  memref<5x4x3x2xf64> to memref<3x2xf64, strided<[2, 1], offset: ?>>

// CHECK-NEXT:  }

// -----

%0 = memref.alloc() : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>>

// Subview with constant offsets, sizes and strides.
%1 = memref.subview %0[0, 2, 0][4, 4, 4][1, 1, 1]
  : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>> to
    memref<4x4x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2 + 8)>>

// CHECK:      Only strided layout attrs implemented

// -----

%m = "test.op"() : () -> memref<2x3xf64, strided<[6, 1], offset: ?>>
%i0, %i1 = "test.op"() : () -> (index, index)

// CHECK:         %m_1 = builtin.unrealized_conversion_cast %m : memref<2x3xf64, strided<[6, 1], offset: ?>> to !riscv.reg
// CHECK-NEXT:    %i0_1 = builtin.unrealized_conversion_cast %i0 : index to !riscv.reg
// CHECK-NEXT:    %i1_1 = builtin.unrealized_conversion_cast %i1 : index to !riscv.reg
// CHECK-NEXT:    %pointer_dim_stride = riscv.li 6
// CHECK-NEXT:    %pointer_dim_offset = riscv.mul %i0_1, %pointer_dim_stride
// CHECK-NEXT:    %pointer_offset = riscv.add %pointer_dim_offset, %i1_1
// CHECK-NEXT:    %bytes_per_element = riscv.li 8
// CHECK-NEXT:    %scaled_pointer_offset = riscv.mul %pointer_offset, %bytes_per_element
// CHECK-NEXT:    %offset_pointer = riscv.add %m_1, %scaled_pointer_offset
// CHECK-NEXT:    %v = riscv.fld %offset_pointer, 0
// CHECK-NEXT:    %v_1 = builtin.unrealized_conversion_cast %v : !riscv.freg to f64
%v = memref.load %m[%i0, %i1] : memref<2x3xf64, strided<[6, 1], offset: ?>>

// -----

%m = "test.op"() : () -> memref<2x3xf64, strided<[6, 1], offset: ?>>
%v = "test.op"() : () -> f64
%i0, %i1 = "test.op"() : () -> (index, index)

// CHECK:         %v_1 = builtin.unrealized_conversion_cast %v : f64 to !riscv.freg
// CHECK-NEXT:    %m_1 = builtin.unrealized_conversion_cast %m : memref<2x3xf64, strided<[6, 1], offset: ?>> to !riscv.reg
// CHECK-NEXT:    %i0_1 = builtin.unrealized_conversion_cast %i0 : index to !riscv.reg
// CHECK-NEXT:    %i1_1 = builtin.unrealized_conversion_cast %i1 : index to !riscv.reg
// CHECK-NEXT:    %pointer_dim_stride = riscv.li 6
// CHECK-NEXT:    %pointer_dim_offset = riscv.mul %i0_1, %pointer_dim_stride
// CHECK-NEXT:    %pointer_offset = riscv.add %pointer_dim_offset, %i1_1
// CHECK-NEXT:    %bytes_per_element = riscv.li 8
// CHECK-NEXT:    %scaled_pointer_offset = riscv.mul %pointer_offset, %bytes_per_element
// CHECK-NEXT:    %offset_pointer = riscv.add %m_1, %scaled_pointer_offset
// CHECK-NEXT:    riscv.fsd %offset_pointer, %v_1, 0

memref.store %v, %m[%i0, %i1] : memref<2x3xf64, strided<[6, 1], offset: ?>>

// -----

%m = "test.op"() : () -> memref<2xf64, strided<[?]>>
%i0 = "test.op"() : () -> index
%v = memref.load %m[%i0] : memref<2xf64, strided<[?]>>

// CHECK: MemRef memref<2xf64, strided<[?]>> with dynamic stride is not yet implemented
