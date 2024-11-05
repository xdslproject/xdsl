// RUN: xdsl-opt -p convert-memref-to-ptr  --split-input-file --verify-diagnostics %s | filecheck %s

%v, %idx, %arr = "test.op"() : () -> (i32, index, memref<10xi32>)
memref.store %v, %arr[%idx] {"nontemporal" = false} : memref<10xi32>

// CHECK:       %bytes_per_element = ptr_xdsl.type_offset i32 : index
// CHECK-NEXT:  %scaled_pointer_offset = arith.muli %idx, %bytes_per_element : index
// CHECK-NEXT:  %0 = ptr_xdsl.to_ptr %arr : memref<10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %offset_pointer = ptr_xdsl.ptradd %0, %scaled_pointer_offset : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  ptr_xdsl.store %v, %offset_pointer : i32, !ptr_xdsl.ptr

%idx1, %idx2, %arr2 = "test.op"() : () -> (index, index, memref<10x10xi32>)
memref.store %v, %arr2[%idx1, %idx2] {"nontemporal" = false} : memref<10x10xi32>

// CHECK-NEXT:  %idx1, %idx2, %arr2 = "test.op"() : () -> (index, index, memref<10x10xi32>)
// CHECK-NEXT:  %pointer_dim_stride = arith.constant 10 : index
// CHECK-NEXT:  %pointer_dim_offset = arith.muli %idx1, %pointer_dim_stride : index
// CHECK-NEXT:  %pointer_dim_stride_1 = arith.addi %pointer_dim_offset, %idx2 : index
// CHECK-NEXT:  %bytes_per_element_1 = ptr_xdsl.type_offset i32 : index
// CHECK-NEXT:  %scaled_pointer_offset_1 = arith.muli %pointer_dim_stride_1, %bytes_per_element_1 : index
// CHECK-NEXT:  %1 = ptr_xdsl.to_ptr %arr2 : memref<10x10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %offset_pointer_1 = ptr_xdsl.ptradd %1, %scaled_pointer_offset_1 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  ptr_xdsl.store %v, %offset_pointer_1 : i32, !ptr_xdsl.ptr

%lv = memref.load %arr[%idx] {"nontemporal" = false} : memref<10xi32>

// CHECK-NEXT:  %bytes_per_element_2 = ptr_xdsl.type_offset i32 : index
// CHECK-NEXT:  %scaled_pointer_offset_2 = arith.muli %idx, %bytes_per_element_2 : index
// CHECK-NEXT:  %lv = ptr_xdsl.to_ptr %arr : memref<10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %offset_pointer_2 = ptr_xdsl.ptradd %lv, %scaled_pointer_offset_2 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %lv_1 = ptr_xdsl.load %offset_pointer_2 : !ptr_xdsl.ptr -> i32

%lv2 = memref.load %arr2[%idx1, %idx2] {"nontemporal" = false} : memref<10x10xi32>

// CHECK-NEXT:  %pointer_dim_stride_2 = arith.constant 10 : index
// CHECK-NEXT:  %pointer_dim_offset_1 = arith.muli %idx1, %pointer_dim_stride_2 : index
// CHECK-NEXT:  %pointer_dim_stride_3 = arith.addi %pointer_dim_offset_1, %idx2 : index
// CHECK-NEXT:  %bytes_per_element_3 = ptr_xdsl.type_offset i32 : index
// CHECK-NEXT:  %scaled_pointer_offset_3 = arith.muli %pointer_dim_stride_3, %bytes_per_element_3 : index
// CHECK-NEXT:  %lv2 = ptr_xdsl.to_ptr %arr2 : memref<10x10xi32> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %offset_pointer_3 = ptr_xdsl.ptradd %lv2, %scaled_pointer_offset_3 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %lv2_1 = ptr_xdsl.load %offset_pointer_3 : !ptr_xdsl.ptr -> i32

%fv, %farr = "test.op"() : () -> (f64, memref<10xf64>)
memref.store %fv, %farr[%idx] {"nontemporal" = false} : memref<10xf64>

// CHECK-NEXT:  %fv, %farr = "test.op"() : () -> (f64, memref<10xf64>)
// CHECK-NEXT:  %bytes_per_element_4 = ptr_xdsl.type_offset f64 : index
// CHECK-NEXT:  %scaled_pointer_offset_4 = arith.muli %idx, %bytes_per_element_4 : index
// CHECK-NEXT:  %2 = ptr_xdsl.to_ptr %farr : memref<10xf64> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %offset_pointer_4 = ptr_xdsl.ptradd %2, %scaled_pointer_offset_4 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  ptr_xdsl.store %fv, %offset_pointer_4 : f64, !ptr_xdsl.ptr

%flv = memref.load %farr[%idx] {"nontemporal" = false} : memref<10xf64>

// CHECK-NEXT:  %bytes_per_element_5 = ptr_xdsl.type_offset f64 : index
// CHECK-NEXT:  %scaled_pointer_offset_5 = arith.muli %idx, %bytes_per_element_5 : index
// CHECK-NEXT:  %flv = ptr_xdsl.to_ptr %farr : memref<10xf64> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %offset_pointer_5 = ptr_xdsl.ptradd %flv, %scaled_pointer_offset_5 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:  %flv_1 = ptr_xdsl.load %offset_pointer_5 : !ptr_xdsl.ptr -> f64

%fmem = "test.op"() : () -> (memref<f64>)
%flv2 = memref.load %fmem[] {"nontemporal" = false} : memref<f64>

// CHECK-NEXT:  %fmem = "test.op"() : () -> memref<f64>
// CHECK-NEXT:  %flv2 = ptr_xdsl.to_ptr %fmem : memref<f64> -> !ptr_xdsl.ptr
// CHECK-NEXT:  %flv2_1 = ptr_xdsl.load %flv2 : !ptr_xdsl.ptr -> f64

// -----

%fv, %idx, %mstr = "test.op"() : () -> (f64, index, memref<2xf64, strided<[?]>>)
memref.store %fv, %mstr[%idx] {"nontemporal" = false} : memref<2xf64, strided<[?]>>

// CHECK: MemRef memref<2xf64, strided<[?]>> with dynamic stride is not yet implemented

// -----

%fv, %idx, %mstr = "test.op"() : () -> (f64, index, memref<2xf64, affine_map<(d0) -> (d0 * 10)>>)
memref.store %fv, %mstr[%idx] {"nontemporal" = false} : memref<2xf64, affine_map<(d0) -> (d0 * 10)>>

// CHECK: Unsupported layout type affine_map<(d0) -> ((d0 * 10))>
