// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %vector, %i0 = "test.op"() : () -> (vector<4xindex>, index)

  %0 = "vector.insertelement"(%i0, %vector, %i0) : (index, vector<4xindex>, index) -> vector<3xindex>
  // CHECK: Expected dest operand and result to have matching types.

}) : () -> ()