// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %0, %1 = "test.op"() : () -> (memref<4x4xindex>, index)

  %2 = "vector.load"(%0, %1, %1) : (memref<4x4xindex>, index, index) -> vector<2xi32>
  // CHECK: MemRef element type should match the Vector element type.

}) : () -> ()
