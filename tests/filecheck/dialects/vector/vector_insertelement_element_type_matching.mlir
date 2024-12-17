// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %vector, %i0, %f0 = "test.op"() : () -> (vector<4xindex>, index, f64)

  %0 = "vector.insertelement"(%f0, %vector, %i0) : (f64, vector<4xindex>, index) -> vector<4xindex>
  // CHECK: Expected source operand type to match element type of dest operand.

}) : () -> ()
