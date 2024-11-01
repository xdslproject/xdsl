// RUN: xdsl-opt -p convert-memref-to-ptr  --split-input-file --verify-diagnostics %s | filecheck %s


// CHECK:      builtin.module {

%v, %idx, %arr = "test.op"() : () -> (i32, index, memref<10xi32>)
memref.store %v, %arr[%idx] {"nontemporal" = false} : memref<10xi32>

// CHECK-NEXT:  %v, %idx, %arr = "test.op"() : () -> (i32, index, memref<10xi32>)
// CHECK-NEXT:  %bytes_per_element = "opaque_ptr.type_offset"() <{"elem_type" = i32}> : () -> index
// CHECK-NEXT:  %scaled_pointer_offset = arith.muli %idx, %bytes_per_element : index
// CHECK-NEXT:  %0 = memref.to_ptr %arr : memref<10xi32> -> !opaque_ptr.ptr
// CHECK-NEXT:  %offset_pointer = opaque_ptr.ptradd %0, %scaled_pointer_offset : (!opaque_ptr.ptr, index) -> !opaque_ptr.ptr
// CHECK-NEXT:  opaque_ptr.store %v, %offset_pointer : i32, !opaque_ptr.ptr

%idx1, %idx2, %arr2 = "test.op"() : () -> (index, index, memref<10x10xi32>)
memref.store %v, %arr2[%idx1, %idx2] {"nontemporal" = false} : memref<10x10xi32>

// CHECK-NEXT:  %idx1, %idx2, %arr2 = "test.op"() : () -> (index, index, memref<10x10xi32>)
// CHECK-NEXT:  %pointer_dim_stride = arith.constant 10 : index
// CHECK-NEXT:  %pointer_dim_offset = arith.muli %idx1, %pointer_dim_stride : index
// CHECK-NEXT:  %pointer_dim_stride_1 = arith.addi %pointer_dim_offset, %idx2 : index
// CHECK-NEXT:  %bytes_per_element_1 = "opaque_ptr.type_offset"() <{"elem_type" = i32}> : () -> index
// CHECK-NEXT:  %scaled_pointer_offset_1 = arith.muli %pointer_dim_stride_1, %bytes_per_element_1 : index
// CHECK-NEXT:  %1 = memref.to_ptr %arr2 : memref<10x10xi32> -> !opaque_ptr.ptr
// CHECK-NEXT:  %offset_pointer_1 = opaque_ptr.ptradd %1, %scaled_pointer_offset_1 : (!opaque_ptr.ptr, index) -> !opaque_ptr.ptr
// CHECK-NEXT:  opaque_ptr.store %v, %offset_pointer_1 : i32, !opaque_ptr.ptr

%lv = memref.load %arr[%idx] {"nontemporal" = false} : memref<10xi32>

// CHECK-NEXT:  %bytes_per_element_2 = "opaque_ptr.type_offset"() <{"elem_type" = i32}> : () -> index
// CHECK-NEXT:  %scaled_pointer_offset_2 = arith.muli %idx, %bytes_per_element_2 : index
// CHECK-NEXT:  %lv = memref.to_ptr %arr : memref<10xi32> -> !opaque_ptr.ptr
// CHECK-NEXT:  %offset_pointer_2 = opaque_ptr.ptradd %lv, %scaled_pointer_offset_2 : (!opaque_ptr.ptr, index) -> !opaque_ptr.ptr
// CHECK-NEXT:  %lv_1 = opaque_ptr.load %offset_pointer_2 : !opaque_ptr.ptr -> i32

%lv2 = memref.load %arr2[%idx1, %idx2] {"nontemporal" = false} : memref<10x10xi32>

// CHECK-NEXT:  %pointer_dim_stride_2 = arith.constant 10 : index
// CHECK-NEXT:  %pointer_dim_offset_1 = arith.muli %idx1, %pointer_dim_stride_2 : index
// CHECK-NEXT:  %pointer_dim_stride_3 = arith.addi %pointer_dim_offset_1, %idx2 : index
// CHECK-NEXT:  %bytes_per_element_3 = "opaque_ptr.type_offset"() <{"elem_type" = i32}> : () -> index
// CHECK-NEXT:  %scaled_pointer_offset_3 = arith.muli %pointer_dim_stride_3, %bytes_per_element_3 : index
// CHECK-NEXT:  %lv2 = memref.to_ptr %arr2 : memref<10x10xi32> -> !opaque_ptr.ptr
// CHECK-NEXT:  %offset_pointer_3 = opaque_ptr.ptradd %lv2, %scaled_pointer_offset_3 : (!opaque_ptr.ptr, index) -> !opaque_ptr.ptr
// CHECK-NEXT:  %lv2_1 = opaque_ptr.load %offset_pointer_3 : !opaque_ptr.ptr -> i32

%fv, %farr = "test.op"() : () -> (f64, memref<10xf64>)
memref.store %fv, %farr[%idx] {"nontemporal" = false} : memref<10xf64>

// CHECK-NEXT:  %fv, %farr = "test.op"() : () -> (f64, memref<10xf64>)
// CHECK-NEXT:  %bytes_per_element_4 = "opaque_ptr.type_offset"() <{"elem_type" = f64}> : () -> index
// CHECK-NEXT:  %scaled_pointer_offset_4 = arith.muli %idx, %bytes_per_element_4 : index
// CHECK-NEXT:  %2 = memref.to_ptr %farr : memref<10xf64> -> !opaque_ptr.ptr
// CHECK-NEXT:  %offset_pointer_4 = opaque_ptr.ptradd %2, %scaled_pointer_offset_4 : (!opaque_ptr.ptr, index) -> !opaque_ptr.ptr
// CHECK-NEXT:  opaque_ptr.store %fv, %offset_pointer_4 : f64, !opaque_ptr.ptr

%flv = memref.load %farr[%idx] {"nontemporal" = false} : memref<10xf64>

// CHECK-NEXT:  %bytes_per_element_5 = "opaque_ptr.type_offset"() <{"elem_type" = f64}> : () -> index
// CHECK-NEXT:  %scaled_pointer_offset_5 = arith.muli %idx, %bytes_per_element_5 : index
// CHECK-NEXT:  %flv = memref.to_ptr %farr : memref<10xf64> -> !opaque_ptr.ptr
// CHECK-NEXT:  %offset_pointer_5 = opaque_ptr.ptradd %flv, %scaled_pointer_offset_5 : (!opaque_ptr.ptr, index) -> !opaque_ptr.ptr
// CHECK-NEXT:  %flv_1 = opaque_ptr.load %offset_pointer_5 : !opaque_ptr.ptr -> f64

// CHECK-NEXT: }
