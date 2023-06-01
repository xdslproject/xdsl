// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({

  %0, %1, %2, %3 = "test.op"() : () -> (memref<2x2xindex>, vector<2xi32>, vector<2xi1>, index)

  "vector.maskedstore"(%0, %3, %3, %2, %1) : (memref<2x2xindex>, index, index, vector<2xi1>, vector<2xi32>) -> ()
  // CHECK: MemRef element type should match the stored vector type. Obtained types were index and i32.

}) : () -> ()
