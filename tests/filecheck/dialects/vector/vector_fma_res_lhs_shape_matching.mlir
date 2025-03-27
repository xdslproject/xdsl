// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %0, %1 = "test.op"() : () -> (vector<2xindex>, vector<3xindex>)

  %2 = "vector.fma"(%1, %0, %0) : (vector<3xindex>, vector<2xindex>, vector<2xindex>) -> vector<2xindex>
  // CHECK: attribute vector<3xindex> expected from variable 'T', but got vector<2xindex>

}) : () -> ()
