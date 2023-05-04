// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %0, %1, %2 = "test.op"() : () -> (vector<2xindex>, memref<4x4xindex>, index)

  "vector.store"(%0, %1, %2) : (vector<2xindex>, memref<4x4xindex>, index) -> ()
  // CHECK: Expected an index for each dimension.

}) : () -> ()
