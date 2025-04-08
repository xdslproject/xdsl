// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %0, %1 = "test.op"() : () -> (vector<2xindex>, vector<2xi32>)

  %2 = "vector.fma"(%1, %1, %0) : (vector<2xi32>, vector<2xi32>, vector<2xindex>) -> vector<2xi32>
  // CHECK: attribute vector<2xi32> expected from variable 'T', but got vector<2xindex>

}) : () -> ()
