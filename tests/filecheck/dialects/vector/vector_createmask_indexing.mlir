// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %0 = "test.op"() : () -> index

  %1 = "vector.create_mask"(%0) : (index) -> vector<2x2xi1>
  // CHECK: Expected an operand value for each dimension of resultant mask.

}) : () -> ()
