// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %0, %1 = "test.op"() : () -> (vector<2xindex>, vector<2xi32>)

  %2 = "vector.fma"(%0, %1, %1) : (vector<2xindex>, vector<2xi32>, vector<2xi32>) -> vector<2xi32>
  // CHECK: attribute vector<2xindex> expected from variable 'T', but got vector<2xi32>

}) : () -> ()
