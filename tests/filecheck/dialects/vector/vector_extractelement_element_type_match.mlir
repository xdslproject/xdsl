// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %vector, %i0= "test.op"() : () -> (vector<4xindex>, index)

  %0 = "vector.extractelement"(%vector, %i0) : (vector<4xindex>, index) -> f64
  // CHECK: Expected result type to match element type of vector operand.

}) : () -> ()
