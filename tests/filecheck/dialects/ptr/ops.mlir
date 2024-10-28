// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

builtin.module {
  %p, %idx = "test.op"() : () -> (!ptr.ptr, index)

  // CHECK: %r0 = ptr.ptradd %p, %idx : (!ptr.ptr, index) -> !ptr.ptr
  %r0 = ptr.ptradd %p, %idx : (!ptr.ptr, index) -> !ptr.ptr
}
