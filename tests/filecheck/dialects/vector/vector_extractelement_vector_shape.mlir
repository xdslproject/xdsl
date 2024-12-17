// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %vector, %i0 = "test.op"() : () -> (vector<4x4xindex>, index)

  %0 = "vector.extractelement"(%vector, %i0) : (vector<4x4xindex>, index) -> index
  // CHECK: Operation does not verify: Unexpected >1 vector rank.

}) : () -> ()