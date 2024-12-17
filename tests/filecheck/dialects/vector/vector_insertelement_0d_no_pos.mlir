// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %vector, %i0 = "test.op"() : () -> (vector<index>, index)

  %0 = "vector.insertelement"(%i0, %vector, %i0) : (index, vector<index>, index) -> vector<index>
  // CHECK: Expected position to be empty with 0-D vector.

}) : () -> ()
