// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

// CHECK: value = dense<[[(1,2), (3,4)], [(5,6), (7,8)]]> : tensor<2x2xcomplex<i32>>
%complex_tensor_i32 = "test.op"() {"value" = dense<[[(1, 2), (3, 4)], [(5, 6), (7, 8)]]> : tensor<2x2xcomplex<i32>>} : () -> tensor<2x2xcomplex<i32>>
// CHECK-NEXT: value = dense<[[(1.000000e+00,2.000000e+00), (3.000000e+00,4.000000e+00)], [(5.000000e+00,6.000000e+00), (7.000000e+00,8.000000e+00)]]> : tensor<2x2xcomplex<f32>>
%complex_tensor_f32 = "test.op"() {"value" = dense<[[(1.0, 2.0), (3.0, 4.0)], [(5.0, 6.0), (7.0, 8.0)]]> : tensor<2x2xcomplex<f32>>} : () -> tensor<2x2xcomplex<f32>>
// CHECK-NEXT: value = dense<[(true,false), (false,false)]>
%complex_tensor_i1 = "test.op"() {"value" = dense<[(true, false), (false, false)]> : tensor<2xcomplex<i1>>} : () -> tensor<2xcomplex<i1>>
