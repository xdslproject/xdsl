// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %vector, %i0 = "test.op"() : () -> (vector<4x4xindex>, index)

  %0 = "vector.insertelement"(%i0, %vector, %i0) : (index, vector<4x4xindex>, index) -> vector<4x4xindex>
  // CHECK: Operation does not verify: Unexpected >1 vector rank.

}) : () -> ()
