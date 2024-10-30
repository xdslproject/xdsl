// RUN: xdsl-opt -p convert-memref-to-ptr  --split-input-file --verify-diagnostics %s | filecheck %s


// CHECK:      builtin.module {

%v, %idx, %arr = "test.op"() : () -> (i32, index, memref<10xi32>)
memref.store %v, %arr[%idx] {"nontemporal" = false} : memref<10xi32>

// CHECK-NEXT:  %v, %idx, %arr = "test.op"() : () -> (i32, index, memref<10xi32>)
// CHECK-NEXT:  %0 = arith.constant 0 : i32
// CHECK-NEXT:  %1 = ptr.type_offset %0 : i32 -> index
// CHECK-NEXT:  %2 = arith.muli %idx, %1 : index
// CHECK-NEXT:  %3 = memref.to_ptr %arr : memref<10xi32> -> !ptr.ptr
// CHECK-NEXT:  %4 = ptr.ptradd %3, %2 : (!ptr.ptr, index) -> !ptr.ptr
// CHECK-NEXT:  ptr.store %v, %4 : i32, !ptr.ptr

%idx1, %idx2, %arr2 = "test.op"() : () -> (index, index, memref<10x10xi32>)
memref.store %v, %arr2[%idx1, %idx2] {"nontemporal" = false} : memref<10x10xi32>

// CHECK-NEXT:  %idx1, %idx2, %arr2 = "test.op"() : () -> (index, index, memref<10x10xi32>)
// CHECK-NEXT:  %5 = arith.constant 10 : index
// CHECK-NEXT:  %6 = arith.muli %idx1, %5 : index
// CHECK-NEXT:  %7 = arith.addi %6, %idx2 : index
// CHECK-NEXT:  %8 = arith.constant 0 : i32
// CHECK-NEXT:  %9 = ptr.type_offset %8 : i32 -> index
// CHECK-NEXT:  %10 = arith.muli %7, %9 : index
// CHECK-NEXT:  %11 = memref.to_ptr %arr2 : memref<10x10xi32> -> !ptr.ptr
// CHECK-NEXT:  %12 = ptr.ptradd %11, %10 : (!ptr.ptr, index) -> !ptr.ptr
// CHECK-NEXT:  ptr.store %v, %12 : i32, !ptr.ptr

%lv = memref.load %arr[%idx] {"nontemporal" = false} : memref<10xi32>

// CHECK-NEXT:  %lv = arith.constant 0 : i32
// CHECK-NEXT:  %lv_1 = ptr.type_offset %lv : i32 -> index
// CHECK-NEXT:  %lv_2 = arith.muli %idx, %lv_1 : index
// CHECK-NEXT:  %lv_3 = memref.to_ptr %arr : memref<10xi32> -> !ptr.ptr
// CHECK-NEXT:  %lv_4 = ptr.ptradd %lv_3, %lv_2 : (!ptr.ptr, index) -> !ptr.ptr
// CHECK-NEXT:  %lv_5 = ptr.load %lv_4 : !ptr.ptr -> i32

%lv2 = memref.load %arr2[%idx1, %idx2] {"nontemporal" = false} : memref<10x10xi32>

// CHECK-NEXT:  %lv2 = arith.constant 10 : index
// CHECK-NEXT:  %lv2_1 = arith.muli %idx1, %lv2 : index
// CHECK-NEXT:  %lv2_2 = arith.addi %lv2_1, %idx2 : index
// CHECK-NEXT:  %lv2_3 = arith.constant 0 : i32
// CHECK-NEXT:  %lv2_4 = ptr.type_offset %lv2_3 : i32 -> index
// CHECK-NEXT:  %lv2_5 = arith.muli %lv2_2, %lv2_4 : index
// CHECK-NEXT:  %lv2_6 = memref.to_ptr %arr2 : memref<10x10xi32> -> !ptr.ptr
// CHECK-NEXT:  %lv2_7 = ptr.ptradd %lv2_6, %lv2_5 : (!ptr.ptr, index) -> !ptr.ptr
// CHECK-NEXT:  %lv2_8 = ptr.load %lv2_7 : !ptr.ptr -> i32

// CHECK-NEXT: }
