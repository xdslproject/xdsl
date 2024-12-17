// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %vector, %i0 = "test.op"() : () -> (vector<1xindex>, index)

  %1 = "vector.extractelement"(%vector) : (vector<1xindex>) -> index
  // CHECK: Expected position for 1-D vector.

}) : () -> ()