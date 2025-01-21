// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %0 = "test.op"() : () -> index
  %1 = "vector.broadcast"(%0) : (index) -> vector<2xi32>
  // CHECK: Source operand and result vector must have the same element type.

}) : () -> ()
