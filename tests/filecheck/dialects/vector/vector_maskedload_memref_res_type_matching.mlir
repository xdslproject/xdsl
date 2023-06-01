// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %0, %1, %2, %3 = "test.op"() : () -> (memref<2x2xindex>, vector<2xindex>, vector<2xi1>, index)

  %4 = "vector.maskedload"(%0, %3, %3, %2, %1) : (memref<2x2xindex>, index, index, vector<2xi1>, vector<2xindex>) -> vector<2xi32>
  // CHECK: MemRef element type should match the result vector and passthrough vector element type. Found different element types for memref and result.

}) : () -> ()
