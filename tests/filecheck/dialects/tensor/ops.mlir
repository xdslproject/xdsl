// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%index, %tensor = "test.op"() : () -> (index, tensor<?x?xf32>)
%dim1 = "tensor.dim"(%tensor, %index): (tensor<?x?xf32>, index) -> index
%dim2 = "tensor.dim"(%tensor, %index) {"hello" = "world"}: (tensor<?x?xf32>, index) -> index
%cast1 = "tensor.cast"(%tensor) : (tensor<?x?xf32>) -> tensor<4x4xf32>
%cast2 = "tensor.cast"(%tensor) {"hello" = "world"} : (tensor<?x?xf32>) -> tensor<4x4xf32>

// CHECK: builtin.module {
// CHECK-NEXT:   %index, %tensor = "test.op"() : () -> (index, tensor<?x?xf32>)
// CHECK-NEXT:   %dim1 = tensor.dim %tensor, %index : tensor<?x?xf32>
// CHECK-NEXT:   %dim2 = tensor.dim {"hello" = "world"} %tensor, %index : tensor<?x?xf32>
// CHECK-NEXT:   %cast1 = tensor.cast %tensor : tensor<?x?xf32> to tensor<4x4xf32>
// CHECK-NEXT:   %cast2 = tensor.cast %tensor {"hello" = "world"} : tensor<?x?xf32> to tensor<4x4xf32>
// CHECK-NEXT: }

// CHECK-GENERIC: "builtin.module"() ({
// CHECK-GENERIC-NEXT:   %index, %tensor = "test.op"() : () -> (index, tensor<?x?xf32>)
// CHECK-GENERIC-NEXT:   %dim1 = "tensor.dim"(%tensor, %index) : (tensor<?x?xf32>, index) -> index
// CHECK-GENERIC-NEXT:   %dim2 = "tensor.dim"(%tensor, %index) {"hello" = "world"} : (tensor<?x?xf32>, index) -> index
// CHECK-GENERIC-NEXT:   %cast1 = "tensor.cast"(%tensor) : (tensor<?x?xf32>) -> tensor<4x4xf32>
// CHECK-GENERIC-NEXT:   %cast2 = "tensor.cast"(%tensor) {"hello" = "world"} : (tensor<?x?xf32>) -> tensor<4x4xf32>
// CHECK-GENERIC-NEXT: }) : () -> ()


