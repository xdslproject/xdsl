// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

%t1 = tensor.empty() : tensor<2x3xf32>
%t2 = tensor.empty() : tensor<2xf32>
%i1 = "test.op"() : () -> (index)
%t3 = tensor.empty(%i1) : tensor<?xf32>
%src = "test.op"() {"value" = dense<1.000000e-01> : tensor<4x1xf32>} : () -> tensor<4x1xf32>
%shape = "test.op"() : () -> (tensor<1xi32>)
%t4 = tensor.reshape %src(%shape) : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>
%inserted_slice = "tensor.insert_slice"(%t2, %t1) <{"static_offsets" = array<i64: 0, 1>, "static_sizes" = array<i64: 1, 2>, "static_strides" = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (tensor<2xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
%extracted_slice = "tensor.extract_slice"(%t1) <{"static_offsets" = array<i64: 0, 1>, "static_sizes" = array<i64: 1, 2>, "static_strides" = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<2x3xf32>) -> tensor<2xf32>
%dim1 = "tensor.dim"(%t1, %i1): (tensor<2x3xf32>, index) -> index
%dim2 = "tensor.dim"(%t1, %i1) {"hello" = "world"}: (tensor<2x3xf32>, index) -> index
%cast1 = "tensor.cast"(%t1) : (tensor<2x3xf32>) -> tensor<?x?xf32>
%cast2 = "tensor.cast"(%t1) {"hello" = "world"} : (tensor<2x3xf32>) -> tensor<?x?xf32>
%big_tensor = tensor.empty() : tensor<2x3x2x3xf32>
%collapsed = tensor.collapse_shape %big_tensor [[0, 1], [2, 3]] : tensor<2x3x2x3xf32> into tensor<6x6xf32>
%extracted = tensor.extract %t2[%i1] : tensor<2xf32>
%tensor_with_ins = tensor.insert %extracted into %t2[%i1] : tensor<2xf32>
%fromelements = tensor.from_elements %i1, %i1: tensor<2xindex>
%index, %index1, %tensor = "test.op"() : () -> (index, index, tensor<?x?xf32>)
%expanded = tensor.expand_shape %tensor [[0, 1, 2], [3]] output_shape [%dim1, 1, 1, %dim2] : tensor<?x?xf32> into tensor<?x1x1x?xf32>
%expanded_with_attr_dict = tensor.expand_shape %tensor [[0, 1, 2], [3]] output_shape [%dim1, 1, 1, %dim2] {test_attr = 42 : i8} : tensor<?x?xf32> into tensor<?x1x1x?xf32>
%expanded_generic = "tensor.expand_shape"(%tensor, %dim1, %dim2) <{reassociation = [[0 : i64, 1 : i64, 2 : i64], [3 : i64]], static_output_shape = array<i64: -9223372036854775808, 1, 1, -9223372036854775808>}> : (tensor<?x?xf32>, index, index) -> tensor<?x1x1x?xf32>

%s = "test.op"() : () -> (f32)
%v = tensor.splat %s : tensor<8xf32>
%v2 = tensor.splat %s[%index, %index1] : tensor<?x8x?xf32>


// CHECK:       module {
// CHECK-NEXT:  %0 = tensor.empty() :  tensor<2x3xf32>
// CHECK-NEXT:  %1 = tensor.empty() : tensor<2xf32>
// CHECK-NEXT:  %2 = "test.op"() : () -> index
// CHECK-NEXT:  %3 = tensor.empty(%2) : tensor<?xf32>
// CHECK-NEXT:  %4 = "test.op"() {value = dense<1.000000e-01> : tensor<4x1xf32>} : () -> tensor<4x1xf32>
// CHECK-NEXT:  %5 = "test.op"() : () -> tensor<1xi32>
// CHECK-NEXT:  %reshape = tensor.reshape %4(%5) : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>
// CHECK-NEXT:  %inserted_slice = tensor.insert_slice %1 into %0[0, 1] [1, 2] [1, 1] : tensor<2xf32> into tensor<2x3xf32>
// CHECK-NEXT:  %extracted_slice = tensor.extract_slice %0[0, 1] [1, 2] [1, 1] : tensor<2x3xf32> to tensor<2xf32>
// CHECK-NEXT:  %{{.*}} = tensor.dim %{{.*}}, %{{.*}} : tensor<2x3xf32>
// CHECK-NEXT:  %{{.*}} = tensor.dim {hello = "world"} %{{.*}}, %{{.*}} : tensor<2x3xf32>
// CHECK-NEXT:  %{{.*}} = tensor.cast %{{.*}} : tensor<2x3xf32> to tensor<?x?xf32>
// CHECK-NEXT:  %{{.*}} = tensor.cast %{{.*}} {hello = "world"} : tensor<2x3xf32> to tensor<?x?xf32>
// CHECK-NEXT:  %6 = tensor.empty() : tensor<2x3x2x3xf32>
// CHECK-NEXT:  %collapsed = tensor.collapse_shape %6 [[0, 1], [2, 3]] : tensor<2x3x2x3xf32> into tensor<6x6xf32>
// CHECK-NEXT:  %extracted = tensor.extract %1[%2] : tensor<2xf32>
// CHECK-NEXT:  %{{.*}} = tensor.insert %extracted into %1[%2] : tensor<2xf32>
// CHECK-NEXT:  %{{.*}} = tensor.from_elements %2, %2 : tensor<2xindex>
// CHECK-NEXT:  %7:3 = "test.op"() : () -> (index, index, tensor<?x?xf32>)
// CHECK-NEXT:  %expanded = tensor.expand_shape %7#2 [[0, 1, 2], [3]] output_shape [%dim, 1, 1, %dim_0] : tensor<?x?xf32> into tensor<?x1x1x?xf32>
// CHECK-NEXT:  %expanded_2 = tensor.expand_shape %7#2 [[0, 1, 2], [3]] output_shape [%dim, 1, 1, %dim_0] {test_attr = 42 : i8} : tensor<?x?xf32> into tensor<?x1x1x?xf32>
// CHECK-NEXT:  %expanded_3 = tensor.expand_shape %7#2 [[0, 1, 2], [3]] output_shape [%dim, 1, 1, %dim_0] : tensor<?x?xf32> into tensor<?x1x1x?xf32>
// CHECK-NEXT:  %8 = "test.op"() : () -> f32
// CHECK-NEXT:  %{{.*}} = tensor.splat %8 : tensor<8xf32>
// CHECK-NEXT:  %{{.*}} = tensor.splat %8[%7#0, %7#1] : tensor<?x8x?xf32>
// CHECK-NEXT: }
