// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %0, %1 = "test.op"() : () -> (memref<4x4xindex>, index)

  %2 = "vector.load"(%0, %1) : (memref<4x4xindex>, index) -> vector<2xindex>
  // CHECK: Expected an index for each dimension.

}) : () -> ()
