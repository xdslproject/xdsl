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

%vcast = "test.op"() : () -> (memref<10xi32>)
%cast = "memref.cast"(%vcast) : (memref<10xi32>) -> memref<?xi32>
%cast2 = "memref.cast"(%cast) : (memref<?xi32>) -> memref<10xi32>
"test.op"(%cast2) : (memref<10xi32>) -> ()

// CHECK-NEXT:  %vcast = "test.op"() : () -> memref<10xi32>
// CHECK-NEXT: "test.op"(%vcast) : (memref<10xi32>) -> ()

%fmemcast = "test.op"() : () -> (memref<f64>)
%fmemcast2 = memref.reinterpret_cast %fmemcast to offset: [0], sizes: [5, 2], strides: [2, 1] : memref<f64> to memref<5x2xf64> 

// CHECK-NEXT:  %fmemcast = "test.op"() : () -> memref<f64>
// CHECK-NEXT:  %fmemcast2 = ptr_xdsl.to_ptr %fmemcast : memref<f64> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %fmemcast2_1 = builtin.unrealized_conversion_cast %fmemcast2 : !ptr_xdsl.ptr to memref<5x2xf64>

// -----

%fv, %idx, %mstr = "test.op"() : () -> (f64, index, memref<2xf64, strided<[?]>>)
memref.store %fv, %mstr[%idx] {"nontemporal" = false} : memref<2xf64, strided<[?]>>

// CHECK: MemRef memref<2xf64, strided<[?]>> with dynamic stride is not yet implemented

// -----

%fv, %idx, %mstr = "test.op"() : () -> (f64, index, memref<2xf64, affine_map<(d0) -> (d0 * 10)>>)
memref.store %fv, %mstr[%idx] {"nontemporal" = false} : memref<2xf64, affine_map<(d0) -> (d0 * 10)>>

// CHECK: Unsupported layout type affine_map<(d0) -> ((d0 * 10))>
