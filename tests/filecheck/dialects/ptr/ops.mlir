// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

builtin.module {
  %p, %idx, %v = "test.op"() : () -> (!opaque_ptr.ptr, index, i32)

  // CHECK: %r0 = opaque_ptr.ptradd %p, %idx : (!opaque_ptr.ptr, index) -> !opaque_ptr.ptr
  %r0 = opaque_ptr.ptradd %p, %idx : (!opaque_ptr.ptr, index) -> !opaque_ptr.ptr
  
  // CHECK-NEXT: %r1 = "opaque_ptr.type_offset"() <{"elem_type" = i32}> : () -> index
  %r1 = "opaque_ptr.type_offset"() <{"elem_type" = i32}> : () -> index
  
  // CHECK-NEXT: opaque_ptr.store %v, %p : i32, !opaque_ptr.ptr
  opaque_ptr.store %v, %p : i32, !opaque_ptr.ptr
  
  // CHECK-NEXT: opaque_ptr.load %p : !opaque_ptr.ptr -> i32
  %r3 = opaque_ptr.load %p : !opaque_ptr.ptr -> i32
}
