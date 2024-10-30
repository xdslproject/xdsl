// RUN: xdsl-opt -p convert-memref-to-ptr  --split-input-file --verify-diagnostics %s | filecheck %s


// CHECK:      builtin.module {

%v, %idx, %arr = "test.op"() : () -> (i32, index, memref<10xi32>)
memref.store %v, %arr[%idx] {"nontemporal" = false} : memref<10xi32>

// CHECK-NEXT:  %v, %idx, %arr = "test.op"() : () -> (i32, index, memref<10xi32>)
// CHECK-NEXT:  %0 = "ptr.type_offset"() <{"elem_type" = i32}> : () -> index
// CHECK-NEXT:  %1 = arith.muli %idx, %0 : index
// CHECK-NEXT:  %2 = memref.to_ptr %arr : memref<10xi32> -> !ptr.ptr
// CHECK-NEXT:  %3 = ptr.ptradd %2, %1 : (!ptr.ptr, index) -> !ptr.ptr
// CHECK-NEXT:  ptr.store %v, %3 : i32, !ptr.ptr

%idx1, %idx2, %arr2 = "test.op"() : () -> (index, index, memref<10x10xi32>)
memref.store %v, %arr2[%idx1, %idx2] {"nontemporal" = false} : memref<10x10xi32>

// CHECK-NEXT:  %idx1, %idx2, %arr2 = "test.op"() : () -> (index, index, memref<10x10xi32>)
// CHECK-NEXT:  %4 = arith.constant 10 : index
// CHECK-NEXT:  %5 = arith.muli %idx1, %4 : index
// CHECK-NEXT:  %6 = arith.addi %5, %idx2 : index
// CHECK-NEXT:  %7 = "ptr.type_offset"() <{"elem_type" = i32}> : () -> index
// CHECK-NEXT:  %8 = arith.muli %6, %7 : index
// CHECK-NEXT:  %9 = memref.to_ptr %arr2 : memref<10x10xi32> -> !ptr.ptr
// CHECK-NEXT:  %10 = ptr.ptradd %9, %8 : (!ptr.ptr, index) -> !ptr.ptr
// CHECK-NEXT:  ptr.store %v, %10 : i32, !ptr.ptr

%lv = memref.load %arr[%idx] {"nontemporal" = false} : memref<10xi32>

// CHECK-NEXT:  %lv = "ptr.type_offset"() <{"elem_type" = i32}> : () -> index
// CHECK-NEXT:  %lv_1 = arith.muli %idx, %lv : index
// CHECK-NEXT:  %lv_2 = memref.to_ptr %arr : memref<10xi32> -> !ptr.ptr
// CHECK-NEXT:  %lv_3 = ptr.ptradd %lv_2, %lv_1 : (!ptr.ptr, index) -> !ptr.ptr
// CHECK-NEXT:  %lv_4 = ptr.load %lv_3 : !ptr.ptr -> i32

%lv2 = memref.load %arr2[%idx1, %idx2] {"nontemporal" = false} : memref<10x10xi32>

// CHECK-NEXT:  %lv2 = arith.constant 10 : index
// CHECK-NEXT:  %lv2_1 = arith.muli %idx1, %lv2 : index
// CHECK-NEXT:  %lv2_2 = arith.addi %lv2_1, %idx2 : index
// CHECK-NEXT:  %lv2_3 = "ptr.type_offset"() <{"elem_type" = i32}> : () -> index
// CHECK-NEXT:  %lv2_4 = arith.muli %lv2_2, %lv2_3 : index
// CHECK-NEXT:  %lv2_5 = memref.to_ptr %arr2 : memref<10x10xi32> -> !ptr.ptr
// CHECK-NEXT:  %lv2_6 = ptr.ptradd %lv2_5, %lv2_4 : (!ptr.ptr, index) -> !ptr.ptr
// CHECK-NEXT:  %lv2_7 = ptr.load %lv2_6 : !ptr.ptr -> i32

// CHECK-NEXT: }
