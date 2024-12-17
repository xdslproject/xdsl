// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %vector, %i0 = "test.op"() : () -> (vector<1xindex>, index)

  %1 = "vector.insertelement"(%i0, %vector) : (index, vector<1xindex>) -> vector<1xindex>
  // CHECK: Expected position for 1-D vector.

}) : () -> ()
