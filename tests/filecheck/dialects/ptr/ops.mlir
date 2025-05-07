// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

%p, %idx, %v, %m = "test.op"() : () -> (!ptr_xdsl.ptr, index, i32, memref<10xi32>)

// CHECK: %r0 = ptr_xdsl.ptradd %p, %idx : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
%r0 = ptr_xdsl.ptradd %p, %idx : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr

// CHECK-NEXT: %r1 = ptr_xdsl.type_offset i32 : index
%r1 = ptr_xdsl.type_offset i32 : index

// CHECK-NEXT: ptr_xdsl.store %v, %p : i32, !ptr_xdsl.ptr
ptr_xdsl.store %v, %p : i32, !ptr_xdsl.ptr

// CHECK-NEXT: ptr_xdsl.load %p : !ptr_xdsl.ptr -> i32
%r3 = ptr_xdsl.load %p : !ptr_xdsl.ptr -> i32

// CHECK-NEXT: %pm = ptr_xdsl.to_ptr %m : memref<10xi32> -> !ptr_xdsl.ptr
%pm = ptr_xdsl.to_ptr %m : memref<10xi32> -> !ptr_xdsl.ptr

// CHECK-NEXT: %mp = ptr_xdsl.from_ptr %p : !ptr_xdsl.ptr -> memref<10xi32>
%mp = ptr_xdsl.from_ptr %p : !ptr_xdsl.ptr -> memref<10xi32>
