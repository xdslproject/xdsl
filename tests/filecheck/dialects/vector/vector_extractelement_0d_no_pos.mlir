// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %vector, %i0 = "test.op"() : () -> (vector<index>, index)

  %0 = "vector.extractelement"(%vector, %i0) : (vector<index>, index) -> index
  // CHECK: Expected position to be empty with 0-D vector.

}) : () -> ()
