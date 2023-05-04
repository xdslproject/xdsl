// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %0, %1 = "test.op"() : () -> (vector<2xindex>, vector<2xi32>)

  %2 = "vector.fma"(%0, %1, %1) : (vector<2xindex>, vector<2xi32>, vector<2xi32>) -> vector<2xi32>
  // CHECK: Result vector type must match with all source vectors. Found different types for result vector and lhs vector.

}) : () -> ()
