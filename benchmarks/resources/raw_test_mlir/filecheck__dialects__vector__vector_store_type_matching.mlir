// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %0, %1, %2 = "test.op"() : () -> (vector<2xi32>, memref<4x4xindex>, index)

  "vector.store"(%0, %1, %2) : (vector<2xi32>, memref<4x4xindex>, index) -> ()
  // CHECK: MemRef element type should match the Vector element type.

}) : () -> ()
