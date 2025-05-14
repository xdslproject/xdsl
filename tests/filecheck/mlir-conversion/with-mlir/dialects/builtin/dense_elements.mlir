// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

// CHECK: value = dense<[[(1,2), (3,4)], [(5,6), (7,8)]]> : tensor<2x2xcomplex<i32>>
%complex_tensor = "test.op"() {"value" = dense<[[(1, 2), (3, 4)], [(5, 6), (7, 8)]]> : tensor<2x2xcomplex<i32>>} : () -> tensor<2x2xcomplex<i32>>
