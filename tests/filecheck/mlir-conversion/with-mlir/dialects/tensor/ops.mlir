// RUN: XDSL_ROUNDTRIP

%t1 = "tensor.empty"() : tensor<2x3xf32>
// CHECK:   %t1 = tensor.empty() : tensor<2x3xf32>
