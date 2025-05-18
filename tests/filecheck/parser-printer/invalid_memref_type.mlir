// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

// CHECK: Invalid offset -2 for strided layout, which must be nonnegative.
"test.op"() {neg_strides = memref<64xf64, strided<[5, 10], offset: -2>>} : () -> ()

// -----

// CHECK: Invalid stride -10 at index 1 for strided layout, which must be positive.
"test.op"() {neg_strides = memref<64xf64, strided<[5, -10], offset: 2>>} : () -> ()
