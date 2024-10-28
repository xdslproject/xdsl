// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

builtin.module {
  %p, %idx, %v = "test.op"() : () -> (!ptr.ptr, index, i32)

  // CHECK: %r0 = ptr.ptradd %p, %idx : (!ptr.ptr, index) -> !ptr.ptr
  %r0 = ptr.ptradd %p, %idx : (!ptr.ptr, index) -> !ptr.ptr
  
  // CHECK-NEXT: %r1 = ptr.type_offset %v : i32 -> index
  %r1 = ptr.type_offset %v : i32 -> index
  
  // CHECK-NEXT: ptr.store %v, %p : i32, !ptr.ptr
  ptr.store %v, %p : i32, !ptr.ptr
  
  // CHECK-NEXT: ptr.load %p : !ptr.ptr -> i32
  %r3 = ptr.load %p : !ptr.ptr -> i32
}
