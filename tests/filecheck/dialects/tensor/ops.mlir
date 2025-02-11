// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%index, %index1, %tensor = "test.op"() : () -> (index, index, tensor<?x?xf32>)
%dim1 = "tensor.dim"(%tensor, %index): (tensor<?x?xf32>, index) -> index
%dim2 = "tensor.dim"(%tensor, %index) {"hello" = "world"}: (tensor<?x?xf32>, index) -> index
%cast1 = "tensor.cast"(%tensor) : (tensor<?x?xf32>) -> tensor<4x4xf32>
%cast2 = "tensor.cast"(%tensor) {"hello" = "world"} : (tensor<?x?xf32>) -> tensor<4x4xf32>
%extract1 = tensor.extract %tensor[%index, %index1] : tensor<?x?xf32>
%insert1 = tensor.insert %extract1 into %tensor[%index, %index1] : tensor<?x?xf32>


// CHECK: builtin.module {
// CHECK-NEXT:   %index, %index1, %tensor = "test.op"() : () -> (index, index, tensor<?x?xf32>)
// CHECK-NEXT:   %dim1 = tensor.dim %tensor, %index : tensor<?x?xf32>
// CHECK-NEXT:   %dim2 = tensor.dim {hello = "world"} %tensor, %index : tensor<?x?xf32>
// CHECK-NEXT:   %cast1 = tensor.cast %tensor : tensor<?x?xf32> to tensor<4x4xf32>
// CHECK-NEXT:   %cast2 = tensor.cast %tensor {hello = "world"} : tensor<?x?xf32> to tensor<4x4xf32>
// CHECK-NEXT:   %extract1 = tensor.extract %tensor[%index, %index1] : tensor<?x?xf32>
// CHECK-NEXT:   %insert1 = tensor.insert %extract1 into %tensor[%index, %index1] : tensor<?x?xf32>
// CHECK-NEXT: }

// CHECK-GENERIC: "builtin.module"() ({
// CHECK-GENERIC-NEXT:   %index, %index1, %tensor = "test.op"() : () -> (index, index, tensor<?x?xf32>)
// CHECK-GENERIC-NEXT:   %dim1 = "tensor.dim"(%tensor, %index) : (tensor<?x?xf32>, index) -> index
// CHECK-GENERIC-NEXT:   %dim2 = "tensor.dim"(%tensor, %index) {hello = "world"} : (tensor<?x?xf32>, index) -> index
// CHECK-GENERIC-NEXT:   %cast1 = "tensor.cast"(%tensor) : (tensor<?x?xf32>) -> tensor<4x4xf32>
// CHECK-GENERIC-NEXT:   %cast2 = "tensor.cast"(%tensor) {hello = "world"} : (tensor<?x?xf32>) -> tensor<4x4xf32>
// CHECK-GENERIC-NEXT:   %extract1 = "tensor.extract"(%tensor, %index, %index1) : (tensor<?x?xf32>, index, index) -> f32
// CHECK-GENERIC-NEXT:   %insert1 = "tensor.insert"(%extract1, %tensor, %index, %index1) : (f32, tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
// CHECK-GENERIC-NEXT: }) : () -> ()
