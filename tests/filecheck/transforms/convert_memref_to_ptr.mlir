// RUN: xdsl-opt -p convert-memref-to-ptr  --split-input-file --verify-diagnostics %s | filecheck %s

%v, %idx, %arr = "test.op"() : () -> (i32, index, memref<10xi32>)
memref.store %v, %arr[%idx] {"nontemporal" = false} : memref<10xi32>

// CHECK:       %v, %idx, %arr = "test.op"() : () -> (i32, index, memref<10xi32>)
// CHECK-NEXT:  %arr_1 = ptr_xdsl.to_ptr %arr : memref<10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %bytes_per_element = ptr_xdsl.type_offset i32 : index
// CHECK-NEXT:  %scaled_pointer_offset = arith.muli %idx, %bytes_per_element : index
// CHECK-NEXT:  %offset_pointer = ptr_xdsl.ptradd %arr_1, %scaled_pointer_offset : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  ptr_xdsl.store %v, %offset_pointer : i32, !ptr_xdsl.ptr

%idx1, %idx2, %arr2 = "test.op"() : () -> (index, index, memref<10x10xi32>)
memref.store %v, %arr2[%idx1, %idx2] {"nontemporal" = false} : memref<10x10xi32>

// CHECK-NEXT:  %idx1, %idx2, %arr2 = "test.op"() : () -> (index, index, memref<10x10xi32>)
// CHECK-NEXT:  %arr2_1 = ptr_xdsl.to_ptr %arr2 : memref<10x10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %pointer_dim_stride = arith.constant 10 : index
// CHECK-NEXT:  %pointer_dim_offset = arith.muli %idx1, %pointer_dim_stride : index
// CHECK-NEXT:  %pointer_dim_stride_1 = arith.addi %pointer_dim_offset, %idx2 : index
// CHECK-NEXT:  %bytes_per_element_1 = ptr_xdsl.type_offset i32 : index
// CHECK-NEXT:  %scaled_pointer_offset_1 = arith.muli %pointer_dim_stride_1, %bytes_per_element_1 : index
// CHECK-NEXT:  %offset_pointer_1 = ptr_xdsl.ptradd %arr2_1, %scaled_pointer_offset_1 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  ptr_xdsl.store %v, %offset_pointer_1 : i32, !ptr_xdsl.ptr

%lv = memref.load %arr[%idx] {"nontemporal" = false} : memref<10xi32>

// CHECK-NEXT:  %arr_2 = ptr_xdsl.to_ptr %arr : memref<10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %bytes_per_element_2 = ptr_xdsl.type_offset i32 : index
// CHECK-NEXT:  %scaled_pointer_offset_2 = arith.muli %idx, %bytes_per_element_2 : index
// CHECK-NEXT:  %offset_pointer_2 = ptr_xdsl.ptradd %arr_2, %scaled_pointer_offset_2 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %lv = ptr_xdsl.load %offset_pointer_2 : !ptr_xdsl.ptr -> i32

%lv2 = memref.load %arr2[%idx1, %idx2] {"nontemporal" = false} : memref<10x10xi32>

// CHECK-NEXT:  %arr2_2 = ptr_xdsl.to_ptr %arr2 : memref<10x10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %pointer_dim_stride_2 = arith.constant 10 : index
// CHECK-NEXT:  %pointer_dim_offset_1 = arith.muli %idx1, %pointer_dim_stride_2 : index
// CHECK-NEXT:  %pointer_dim_stride_3 = arith.addi %pointer_dim_offset_1, %idx2 : index
// CHECK-NEXT:  %bytes_per_element_3 = ptr_xdsl.type_offset i32 : index
// CHECK-NEXT:  %scaled_pointer_offset_3 = arith.muli %pointer_dim_stride_3, %bytes_per_element_3 : index
// CHECK-NEXT:  %offset_pointer_3 = ptr_xdsl.ptradd %arr2_2, %scaled_pointer_offset_3 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %lv2 = ptr_xdsl.load %offset_pointer_3 : !ptr_xdsl.ptr -> i32

%fv, %farr = "test.op"() : () -> (f64, memref<10xf64>)
memref.store %fv, %farr[%idx] {"nontemporal" = false} : memref<10xf64>

// CHECK-NEXT:  %fv, %farr = "test.op"() : () -> (f64, memref<10xf64>)
// CHECK-NEXT:  %farr_1 = ptr_xdsl.to_ptr %farr : memref<10xf64> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %bytes_per_element_4 = ptr_xdsl.type_offset f64 : index
// CHECK-NEXT:  %scaled_pointer_offset_4 = arith.muli %idx, %bytes_per_element_4 : index
// CHECK-NEXT:  %offset_pointer_4 = ptr_xdsl.ptradd %farr_1, %scaled_pointer_offset_4 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  ptr_xdsl.store %fv, %offset_pointer_4 : f64, !ptr_xdsl.ptr

%flv = memref.load %farr[%idx] {"nontemporal" = false} : memref<10xf64>

// CHECK-NEXT:  %farr_2 = ptr_xdsl.to_ptr %farr : memref<10xf64> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %bytes_per_element_5 = ptr_xdsl.type_offset f64 : index
// CHECK-NEXT:  %scaled_pointer_offset_5 = arith.muli %idx, %bytes_per_element_5 : index
// CHECK-NEXT:  %offset_pointer_5 = ptr_xdsl.ptradd %farr_2, %scaled_pointer_offset_5 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %flv = ptr_xdsl.load %offset_pointer_5 : !ptr_xdsl.ptr -> f64

%fmem = "test.op"() : () -> (memref<f64>)
%flv2 = memref.load %fmem[] {"nontemporal" = false} : memref<f64>

// CHECK-NEXT:  %fmem = "test.op"() : () -> memref<f64>
// CHECK-NEXT:  %fmem_1 = ptr_xdsl.to_ptr %fmem : memref<f64> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %flv2 = ptr_xdsl.load %fmem_1 : !ptr_xdsl.ptr -> f64

%idx3, %offsetmem = "test.op"() : () -> (index, memref<16xf64, strided<[1], offset: 4>>)
%flv3 = memref.load %offsetmem[%idx3] {"nontemporal" = false} : memref<16xf64, strided<[1], offset: 4>>

// CHECK-NEXT:  %idx3, %offsetmem = "test.op"() : () -> (index, memref<16xf64, strided<[1], offset: 4>>
// CHECK-NEXT:  %offsetmem_1 = ptr_xdsl.to_ptr %offsetmem : memref<16xf64, strided<[1], offset: 4>> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %bytes_per_element_6 = ptr_xdsl.type_offset f64 : index
// CHECK-NEXT:  %scaled_pointer_offset_6 = arith.muli %idx3, %bytes_per_element_6 : index
// CHECK-NEXT:  %offset_pointer_6 = ptr_xdsl.ptradd %offsetmem_1, %scaled_pointer_offset_6 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %flv3 = ptr_xdsl.load %offset_pointer_6 : !ptr_xdsl.ptr -> f64

%fv4, %idx4, %mstr4 = "test.op"() : () -> (f32, index, memref<2xf32, strided<[1], offset: ?>>)
memref.store %fv4, %mstr4[%idx4] {"nontemporal" = false} : memref<2xf32, strided<[1], offset: ?>>

// CHECK-NEXT:    %fv4, %idx4, %mstr4 = "test.op"() : () -> (f32, index, memref<2xf32, strided<[1], offset: ?>>)
// CHECK-NEXT:    %mstr4_1 = ptr_xdsl.to_ptr %mstr4 : memref<2xf32, strided<[1], offset: ?>> -> !ptr_xdsl.ptr
// CHECK-NEXT:    %bytes_per_element_7 = ptr_xdsl.type_offset f32 : index
// CHECK-NEXT:    %scaled_pointer_offset_7 = arith.muli %idx4, %bytes_per_element_7 : index
// CHECK-NEXT:    %offset_pointer_7 = ptr_xdsl.ptradd %mstr4_1, %scaled_pointer_offset_7 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:    ptr_xdsl.store %fv4, %offset_pointer_7 : f32, !ptr_xdsl.ptr

%subview1d = memref.subview %arr[5][5][1] : memref<10xi32> to memref<5xi32>

// CHECK-NEXT:  %arr_3 = ptr_xdsl.to_ptr %arr : memref<10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %c5 = arith.constant 5 : index
// CHECK-NEXT:  %bytes_per_element_8 = ptr_xdsl.type_offset i32 : index
// CHECK-NEXT:  %scaled_pointer_offset_8 = arith.muli %c5, %bytes_per_element_8 : index
// CHECK-NEXT:  %offset_pointer_8 = ptr_xdsl.ptradd %arr_3, %scaled_pointer_offset_8 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %subview1d = ptr_xdsl.from_ptr %offset_pointer_8 : !ptr_xdsl.ptr -> memref<5xi32>

%subview2d = memref.subview %arr2[2, 3][5, 4][1, 1] : memref<10x10xi32> to memref<5x4xi32>

// CHECK-NEXT:  %arr2_3 = ptr_xdsl.to_ptr %arr2 : memref<10x10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %c2 = arith.constant 2 : index
// CHECK-NEXT:  %c10 = arith.constant 10 : index
// CHECK-NEXT:  %increment = arith.muli %c10, %c2 : index
// CHECK-NEXT:  %c3 = arith.constant 3 : index
// CHECK-NEXT:  %subview = arith.addi %increment, %c3 : index
// CHECK-NEXT:  %bytes_per_element_9 = ptr_xdsl.type_offset i32 : index
// CHECK-NEXT:  %scaled_pointer_offset_9 = arith.muli %subview, %bytes_per_element_9 : index
// CHECK-NEXT:  %offset_pointer_9 = ptr_xdsl.ptradd %arr2_3, %scaled_pointer_offset_9 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %subview2d = ptr_xdsl.from_ptr %offset_pointer_9 : !ptr_xdsl.ptr -> memref<5x4xi32>

%cast_src = "test.op"() : () -> (memref<10xi32>)
%cast_dst = "memref.cast"(%cast_src) : (memref<10xi32>) -> memref<?xi32>

// CHECK-NEXT:  %cast_src = "test.op"() : () -> memref<10xi32>

"test.op"(%lv, %lv2, %flv, %flv2, %flv3, %subview1d, %subview2d, %cast_dst) : (i32, i32, f64, f64, f64, memref<5xi32>, memref<5x4xi32>, memref<?xi32>) -> ()

// CHECK-NEXT:  "test.op"(%lv, %lv2, %flv, %flv2, %flv3, %subview1d, %subview2d, %cast_src) : (i32, i32, f64, f64, f64, memref<5xi32>, memref<5x4xi32>, memref<10xi32>) -> ()

// -----

// zero offset: just to_ptr + from_ptr, no arithmetic
%zero_src = "test.op"() : () -> (memref<10x2xindex>)
%zero_dst = "memref.reinterpret_cast"(%zero_src) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 5, 4>, static_strides = array<i64: 1, 1>}> : (memref<10x2xindex>) -> memref<5x4xindex, strided<[1, 1]>>

// static offset=5: constant + byte-scale + ptradd
%static_src = "test.op"() : () -> (memref<10xi32>)
%static_dst = "memref.reinterpret_cast"(%static_src) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 5>, static_sizes = array<i64: 5>, static_strides = array<i64: 1>}> : (memref<10xi32>) -> memref<5xi32, strided<[1], offset: 5>>

// dynamic offset: use dynamic operand directly
%dyn_src = "test.op"() : () -> (memref<10xi32>)
%dyn_off = "test.op"() : () -> (index)
%dyn_dst = "memref.reinterpret_cast"(%dyn_src, %dyn_off) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808>, static_sizes = array<i64: 5>, static_strides = array<i64: 1>}> : (memref<10xi32>, index) -> memref<5xi32, strided<[1], offset: ?>>

"test.op"(%zero_dst, %static_dst, %dyn_dst) : (memref<5x4xindex, strided<[1, 1]>>, memref<5xi32, strided<[1], offset: 5>>, memref<5xi32, strided<[1], offset: ?>>) -> ()

// CHECK:       %zero_src = "test.op"() : () -> memref<10x2xindex>
// CHECK-NEXT:  %zero_src_1 = ptr_xdsl.to_ptr %zero_src : memref<10x2xindex> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %zero_dst = ptr_xdsl.from_ptr %zero_src_1 : !ptr_xdsl.ptr -> memref<5x4xindex, strided<[1, 1]>>

// CHECK-NEXT:  %static_src = "test.op"() : () -> memref<10xi32>
// CHECK-NEXT:  %static_src_1 = ptr_xdsl.to_ptr %static_src : memref<10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %c5 = arith.constant 5 : index
// CHECK-NEXT:  %bytes_per_element = ptr_xdsl.type_offset i32 : index
// CHECK-NEXT:  %scaled_pointer_offset = arith.muli %c5, %bytes_per_element : index
// CHECK-NEXT:  %offset_pointer = ptr_xdsl.ptradd %static_src_1, %scaled_pointer_offset : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %static_dst = ptr_xdsl.from_ptr %offset_pointer : !ptr_xdsl.ptr -> memref<5xi32, strided<[1], offset: 5>>

// CHECK-NEXT:  %dyn_src = "test.op"() : () -> memref<10xi32>
// CHECK-NEXT:  %dyn_off = "test.op"() : () -> index
// CHECK-NEXT:  %dyn_src_1 = ptr_xdsl.to_ptr %dyn_src : memref<10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %bytes_per_element_1 = ptr_xdsl.type_offset i32 : index
// CHECK-NEXT:  %scaled_pointer_offset_1 = arith.muli %dyn_off, %bytes_per_element_1 : index
// CHECK-NEXT:  %offset_pointer_1 = ptr_xdsl.ptradd %dyn_src_1, %scaled_pointer_offset_1 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %dyn_dst = ptr_xdsl.from_ptr %offset_pointer_1 : !ptr_xdsl.ptr -> memref<5xi32, strided<[1], offset: ?>>

// CHECK-NEXT:  "test.op"(%zero_dst, %static_dst, %dyn_dst) : (memref<5x4xindex, strided<[1, 1]>>, memref<5xi32, strided<[1], offset: 5>>, memref<5xi32, strided<[1], offset: ?>>) -> ()

// -----

// load from memref with one dynamic dim, strides are all static [8, 1], no memref.dim needed
%idx_a1, %idx_a2, %arr_dyn1 = "test.op"() : () -> (index, index, memref<?x8xf32>)
%lv_dyn1 = memref.load %arr_dyn1[%idx_a1, %idx_a2] {"nontemporal" = false} : memref<?x8xf32>

// CHECK:       %idx_a1, %idx_a2, %arr_dyn1 = "test.op"() : () -> (index, index, memref<?x8xf32>)
// CHECK-NEXT:  %arr_dyn1_1 = ptr_xdsl.to_ptr %arr_dyn1 : memref<?x8xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %pointer_dim_stride = arith.constant 8 : index
// CHECK-NEXT:  %pointer_dim_offset = arith.muli %idx_a1, %pointer_dim_stride : index
// CHECK-NEXT:  %pointer_dim_stride_1 = arith.addi %pointer_dim_offset, %idx_a2 : index
// CHECK-NEXT:  %bytes_per_element = ptr_xdsl.type_offset f32 : index
// CHECK-NEXT:  %scaled_pointer_offset = arith.muli %pointer_dim_stride_1, %bytes_per_element : index
// CHECK-NEXT:  %offset_pointer = ptr_xdsl.ptradd %arr_dyn1_1, %scaled_pointer_offset : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %lv_dyn1 = ptr_xdsl.load %offset_pointer : !ptr_xdsl.ptr -> f32

// load from memref with all dynamic dims, needs memref.dim for stride[0]
%idx_b1, %idx_b2, %arr_dyn2 = "test.op"() : () -> (index, index, memref<?x?xf32>)
%lv_dyn2 = memref.load %arr_dyn2[%idx_b1, %idx_b2] {"nontemporal" = false} : memref<?x?xf32>

// CHECK-NEXT:  %idx_b1, %idx_b2, %arr_dyn2 = "test.op"() : () -> (index, index, memref<?x?xf32>)
// CHECK-NEXT:  %arr_dyn2_1 = ptr_xdsl.to_ptr %arr_dyn2 : memref<?x?xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %dim_idx = arith.constant 1 : index
// CHECK-NEXT:  %0 = memref.dim %arr_dyn2, %dim_idx : memref<?x?xf32>
// CHECK-NEXT:  %pointer_dim_offset_1 = arith.muli %idx_b1, %0 : index
// CHECK-NEXT:  %pointer_dim_stride_2 = arith.addi %pointer_dim_offset_1, %idx_b2 : index
// CHECK-NEXT:  %bytes_per_element_1 = ptr_xdsl.type_offset f32 : index
// CHECK-NEXT:  %scaled_pointer_offset_1 = arith.muli %pointer_dim_stride_2, %bytes_per_element_1 : index
// CHECK-NEXT:  %offset_pointer_1 = ptr_xdsl.ptradd %arr_dyn2_1, %scaled_pointer_offset_1 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %lv_dyn2 = ptr_xdsl.load %offset_pointer_1 : !ptr_xdsl.ptr -> f32

// 3D with middle dynamic dim, exercises stride fold for (static=4, dynamic=dim1)
%i, %j, %k, %m3d = "test.op"() : () -> (index, index, index, memref<2x?x4xf32>)
%lv_3d = memref.load %m3d[%i, %j, %k] {"nontemporal" = false} : memref<2x?x4xf32>

// CHECK-NEXT:  %i, %j, %k, %m3d = "test.op"() : () -> (index, index, index, memref<2x?x4xf32>)
// CHECK-NEXT:  %m3d_1 = ptr_xdsl.to_ptr %m3d : memref<2x?x4xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %dim_idx_1 = arith.constant 1 : index
// CHECK-NEXT:  %1 = memref.dim %m3d, %dim_idx_1 : memref<2x?x4xf32>
// CHECK-NEXT:  %2 = arith.constant 4 : index
// CHECK-NEXT:  %3 = arith.muli %2, %1 : index
// CHECK-NEXT:  %pointer_dim_offset_2 = arith.muli %i, %3 : index
// CHECK-NEXT:  %pointer_dim_stride_3 = arith.constant 4 : index
// CHECK-NEXT:  %pointer_dim_offset_3 = arith.muli %j, %pointer_dim_stride_3 : index
// CHECK-NEXT:  %pointer_dim_stride_4 = arith.addi %pointer_dim_offset_2, %pointer_dim_offset_3 : index
// CHECK-NEXT:  %pointer_dim_stride_5 = arith.addi %pointer_dim_stride_4, %k : index
// CHECK-NEXT:  %bytes_per_element_2 = ptr_xdsl.type_offset f32 : index
// CHECK-NEXT:  %scaled_pointer_offset_2 = arith.muli %pointer_dim_stride_5, %bytes_per_element_2 : index
// CHECK-NEXT:  %offset_pointer_2 = ptr_xdsl.ptradd %m3d_1, %scaled_pointer_offset_2 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %lv_3d = ptr_xdsl.load %offset_pointer_2 : !ptr_xdsl.ptr -> f32

// 1D dynamic memref, stride is always 1, no memref.dim needed
%idx_1d, %arr_1d = "test.op"() : () -> (index, memref<?xf32>)
%lv_1d = memref.load %arr_1d[%idx_1d] {"nontemporal" = false} : memref<?xf32>

// CHECK-NEXT:  %idx_1d, %arr_1d = "test.op"() : () -> (index, memref<?xf32>)
// CHECK-NEXT:  %arr_1d_1 = ptr_xdsl.to_ptr %arr_1d : memref<?xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %bytes_per_element_3 = ptr_xdsl.type_offset f32 : index
// CHECK-NEXT:  %scaled_pointer_offset_3 = arith.muli %idx_1d, %bytes_per_element_3 : index
// CHECK-NEXT:  %offset_pointer_3 = ptr_xdsl.ptradd %arr_1d_1, %scaled_pointer_offset_3 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %lv_1d = ptr_xdsl.load %offset_pointer_3 : !ptr_xdsl.ptr -> f32

// subview with dynamic offset on dynamic-shaped source
%off, %dyn_src = "test.op"() : () -> (index, memref<?x?xf32>)
%subview_dyn = memref.subview %dyn_src[%off, 0][4, 8][1, 1] : memref<?x?xf32> to memref<4x8xf32, strided<[?, 1], offset: ?>>

// CHECK-NEXT:  %off, %dyn_src = "test.op"() : () -> (index, memref<?x?xf32>)
// CHECK-NEXT:  %dim_idx_2 = arith.constant 1 : index
// CHECK-NEXT:  %4 = memref.dim %dyn_src, %dim_idx_2 : memref<?x?xf32>
// CHECK-NEXT:  %dyn_src_1 = ptr_xdsl.to_ptr %dyn_src : memref<?x?xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %increment = arith.muli %4, %off : index
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %subview = arith.addi %increment, %c0 : index
// CHECK-NEXT:  %bytes_per_element_4 = ptr_xdsl.type_offset f32 : index
// CHECK-NEXT:  %scaled_pointer_offset_4 = arith.muli %subview, %bytes_per_element_4 : index
// CHECK-NEXT:  %offset_pointer_4 = ptr_xdsl.ptradd %dyn_src_1, %scaled_pointer_offset_4 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %subview_dyn = ptr_xdsl.from_ptr %offset_pointer_4 : !ptr_xdsl.ptr -> memref<4x8xf32, strided<[?, 1], offset: ?>>

// subview with all-static offsets on dynamic-shaped source
%dyn_src2 = "test.op"() : () -> (memref<?x?xf32>)
%subview_static = memref.subview %dyn_src2[2, 3][4, 8][1, 1] : memref<?x?xf32> to memref<4x8xf32, strided<[?, 1], offset: ?>>

// CHECK-NEXT:  %dyn_src2 = "test.op"() : () -> memref<?x?xf32>
// CHECK-NEXT:  %dim_idx_3 = arith.constant 1 : index
// CHECK-NEXT:  %5 = memref.dim %dyn_src2, %dim_idx_3 : memref<?x?xf32>
// CHECK-NEXT:  %dyn_src2_1 = ptr_xdsl.to_ptr %dyn_src2 : memref<?x?xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %c2 = arith.constant 2 : index
// CHECK-NEXT:  %increment_1 = arith.muli %5, %c2 : index
// CHECK-NEXT:  %c3 = arith.constant 3 : index
// CHECK-NEXT:  %subview_1 = arith.addi %increment_1, %c3 : index
// CHECK-NEXT:  %bytes_per_element_5 = ptr_xdsl.type_offset f32 : index
// CHECK-NEXT:  %scaled_pointer_offset_5 = arith.muli %subview_1, %bytes_per_element_5 : index
// CHECK-NEXT:  %offset_pointer_5 = ptr_xdsl.ptradd %dyn_src2_1, %scaled_pointer_offset_5 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %subview_static = ptr_xdsl.from_ptr %offset_pointer_5 : !ptr_xdsl.ptr -> memref<4x8xf32, strided<[?, 1], offset: ?>>

// covers the `(_, 1)` stride fold and `isinstance(dim_size, int)` branch
%a, %b, %c, %d, %m4d = "test.op"() : () -> (index, index, index, index, memref<?x1x4x?xf32>)
%lv_4d = memref.load %m4d[%a, %b, %c, %d] {"nontemporal" = false} : memref<?x1x4x?xf32>

// CHECK-NEXT:  %a, %b, %c, %d, %m4d = "test.op"() : () -> (index, index, index, index, memref<?x1x4x?xf32>)
// CHECK-NEXT:  %m4d_1 = ptr_xdsl.to_ptr %m4d : memref<?x1x4x?xf32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %dim_idx_4 = arith.constant 3 : index
// CHECK-NEXT:  %6 = memref.dim %m4d, %dim_idx_4 : memref<?x1x4x?xf32>
// CHECK-NEXT:  %7 = arith.constant 4 : index
// CHECK-NEXT:  %8 = arith.muli %6, %7 : index
// CHECK-NEXT:  %pointer_dim_offset_4 = arith.muli %a, %8 : index
// CHECK-NEXT:  %pointer_dim_offset_5 = arith.muli %b, %8 : index
// CHECK-NEXT:  %pointer_dim_stride_6 = arith.addi %pointer_dim_offset_4, %pointer_dim_offset_5 : index
// CHECK-NEXT:  %pointer_dim_offset_6 = arith.muli %c, %6 : index
// CHECK-NEXT:  %pointer_dim_stride_7 = arith.addi %pointer_dim_stride_6, %pointer_dim_offset_6 : index
// CHECK-NEXT:  %pointer_dim_stride_8 = arith.addi %pointer_dim_stride_7, %d : index
// CHECK-NEXT:  %bytes_per_element_6 = ptr_xdsl.type_offset f32 : index
// CHECK-NEXT:  %scaled_pointer_offset_6 = arith.muli %pointer_dim_stride_8, %bytes_per_element_6 : index
// CHECK-NEXT:  %offset_pointer_6 = ptr_xdsl.ptradd %m4d_1, %scaled_pointer_offset_6 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %lv_4d = ptr_xdsl.load %offset_pointer_6 : !ptr_xdsl.ptr -> f32

"test.op"(%lv_dyn1, %lv_dyn2, %lv_3d, %lv_1d, %subview_dyn, %subview_static, %lv_4d) : (f32, f32, f32, f32, memref<4x8xf32, strided<[?, 1], offset: ?>>, memref<4x8xf32, strided<[?, 1], offset: ?>>, f32) -> ()

// CHECK-NEXT:  "test.op"(%lv_dyn1, %lv_dyn2, %lv_3d, %lv_1d, %subview_dyn, %subview_static, %lv_4d) : (f32, f32, f32, f32, memref<4x8xf32, strided<[?, 1], offset: ?>>, memref<4x8xf32, strided<[?, 1], offset: ?>>, f32) -> ()

// -----

%fv, %idx, %mstr = "test.op"() : () -> (f64, index, memref<2xf64, strided<[?]>>)
memref.store %fv, %mstr[%idx] {"nontemporal" = false} : memref<2xf64, strided<[?]>>

// CHECK: MemRef memref<2xf64, strided<[?]>> with dynamic stride is not yet implemented

// -----

// TODO: multi-dim strided layout with dynamic leading stride
%fv2, %idx1_2, %idx2_2, %mstr_2 = "test.op"() : () -> (f64, index, index, memref<4x8xf64, strided<[?, 1]>>)
memref.store %fv2, %mstr_2[%idx1_2, %idx2_2] {"nontemporal" = false} : memref<4x8xf64, strided<[?, 1]>>

// CHECK: MemRef memref<4x8xf64, strided<[?, 1]>> with dynamic stride is not yet implemented

// -----

// TODO: strided layout with all-dynamic strides
%fv3, %idx1_3, %idx2_3, %mstr_3 = "test.op"() : () -> (f64, index, index, memref<4x8xf64, strided<[?, ?]>>)
memref.store %fv3, %mstr_3[%idx1_3, %idx2_3] {"nontemporal" = false} : memref<4x8xf64, strided<[?, ?]>>

// CHECK: MemRef memref<4x8xf64, strided<[?, ?]>> with dynamic stride is not yet implemented

// -----

// TODO: subview whose source has a strided layout with dynamic stride
%sv_src, %sv_off = "test.op"() : () -> (memref<4x8xf64, strided<[?, 1]>>, index)
%sv = memref.subview %sv_src[%sv_off, 0][2, 4][1, 1] : memref<4x8xf64, strided<[?, 1]>> to memref<2x4xf64, strided<[?, 1], offset: ?>>
"test.op"(%sv) : (memref<2x4xf64, strided<[?, 1], offset: ?>>) -> ()

// CHECK: MemRef memref<4x8xf64, strided<[?, 1]>> with dynamic stride is not yet implemented

// -----

%fv, %idx, %mstr = "test.op"() : () -> (f64, index, memref<2xf64, affine_map<(d0) -> (d0 * 10)>>)
memref.store %fv, %mstr[%idx] {"nontemporal" = false} : memref<2xf64, affine_map<(d0) -> (d0 * 10)>>

// CHECK: Unsupported layout type affine_map<(d0) -> ((d0 * 10))>
