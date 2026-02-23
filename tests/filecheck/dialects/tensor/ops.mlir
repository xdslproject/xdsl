// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%index, %index1, %tensor = "test.op"() : () -> (index, index, tensor<?x?xf32>)
%dim1 = "tensor.dim"(%tensor, %index): (tensor<?x?xf32>, index) -> index
%dim2 = "tensor.dim"(%tensor, %index) {"hello" = "world"}: (tensor<?x?xf32>, index) -> index
%cast1 = "tensor.cast"(%tensor) : (tensor<?x?xf32>) -> tensor<4x4xf32>
%cast2 = "tensor.cast"(%tensor) {"hello" = "world"} : (tensor<?x?xf32>) -> tensor<4x4xf32>
%extract1 = tensor.extract %tensor[%index, %index1] : tensor<?x?xf32>
%insert1 = tensor.insert %extract1 into %tensor[%index, %index1] : tensor<?x?xf32>
%fromelements = tensor.from_elements %index, %index1 : tensor<2xindex>
%res_collapse1 = tensor.collapse_shape %tensor [ [0, 1], [2] ] : tensor<?x?xf32> into tensor<4x1xf32>

%expanded = tensor.expand_shape %tensor [[0, 1, 2], [3]] output_shape [%dim1, 1, 1, %dim2] : tensor<?x?xf32> into tensor<?x1x1x?xf32>
%expanded_with_attr_dict = tensor.expand_shape %tensor [[0, 1, 2], [3]] output_shape [%dim1, 1, 1, %dim2] {test_attr = 42 : i8} : tensor<?x?xf32> into tensor<?x1x1x?xf32>
%expanded_generic = "tensor.expand_shape"(%tensor, %dim1, %dim2) <{reassociation = [[0 : i64, 1 : i64, 2 : i64], [3 : i64]], static_output_shape = array<i64: -9223372036854775808, 1, 1, -9223372036854775808>}> : (tensor<?x?xf32>, index, index) -> tensor<?x1x1x?xf32>

%s = "test.op"() : () -> (f32)
%v = tensor.splat %s : tensor<8xf32>
%v2 = tensor.splat %s[%index, %index1] : tensor<?x8x?xf32>

%pval, %ptensor = "test.op"() : () -> (f32, tensor<12x20x20xf32>) 
%padded = tensor.pad %ptensor low[1, %dim1, 2] high[%dim2, 3, 3] {
  ^bb0(%arg0 : index, %arg1 : index, %arg2 : index):
    tensor.yield %pval : f32
  } : tensor<12x20x20xf32> to tensor<?x?x25xf32>

// CHECK: builtin.module {
// CHECK-NEXT:   %index, %index1, %tensor = "test.op"() : () -> (index, index, tensor<?x?xf32>)
// CHECK-NEXT:   %dim1 = tensor.dim %tensor, %index : tensor<?x?xf32>
// CHECK-NEXT:   %dim2 = tensor.dim {hello = "world"} %tensor, %index : tensor<?x?xf32>
// CHECK-NEXT:   %cast1 = tensor.cast %tensor : tensor<?x?xf32> to tensor<4x4xf32>
// CHECK-NEXT:   %cast2 = tensor.cast %tensor {hello = "world"} : tensor<?x?xf32> to tensor<4x4xf32>
// CHECK-NEXT:   %extract1 = tensor.extract %tensor[%index, %index1] : tensor<?x?xf32>
// CHECK-NEXT:   %insert1 = tensor.insert %extract1 into %tensor[%index, %index1] : tensor<?x?xf32>
// CHECK-NEXT:   %fromelements = tensor.from_elements %index, %index1 : tensor<2xindex>
// CHECK-NEXT:   %res_collapse1 = tensor.collapse_shape %tensor [[0 : i64, 1 : i64], [2 : i64]] : tensor<?x?xf32> into tensor<4x1xf32>
// CHECK-NEXT:   %expanded = tensor.expand_shape %tensor [[0 : i64, 1 : i64, 2 : i64], [3 : i64]] output_shape [%dim1, 1, 1, %dim2] : tensor<?x?xf32> into tensor<?x1x1x?xf32>
// CHECK-NEXT:   %expanded_with_attr_dict = tensor.expand_shape %tensor [[0 : i64, 1 : i64, 2 : i64], [3 : i64]] output_shape [%dim1, 1, 1, %dim2] {test_attr = 42 : i8} : tensor<?x?xf32> into tensor<?x1x1x?xf32>
// CHECK-NEXT:   %expanded_generic = tensor.expand_shape %tensor [[0 : i64, 1 : i64, 2 : i64], [3 : i64]] output_shape [%dim1, 1, 1, %dim2] : tensor<?x?xf32> into tensor<?x1x1x?xf32>
// CHECK-NEXT:   %s = "test.op"() : () -> f32
// CHECK-NEXT:   %v = tensor.splat %s : tensor<8xf32>
// CHECK-NEXT:   %v2 = tensor.splat %s[%index, %index1] : tensor<?x8x?xf32>
// CHECK-NEXT:  %pval, %ptensor = "test.op"() : () -> (f32, tensor<12x20x20xf32>) 
// CHECK-NEXT:  %padded = tensor.pad %ptensor low[1, %dim1, 2] high[%dim2, 3, 3] {
// CHECK-NEXT:    ^bb0(%arg0 : index, %arg1 : index, %arg2 : index):
// CHECK-NEXT:      tensor.yield %pval : f32
// CHECK-NEXT:    } : tensor<12x20x20xf32> to tensor<?x?x25xf32>
// CHECK-NEXT: }

// CHECK-GENERIC: "builtin.module"() ({
// CHECK-GENERIC-NEXT:   %index, %index1, %tensor = "test.op"() : () -> (index, index, tensor<?x?xf32>)
// CHECK-GENERIC-NEXT:   %dim1 = "tensor.dim"(%tensor, %index) : (tensor<?x?xf32>, index) -> index
// CHECK-GENERIC-NEXT:   %dim2 = "tensor.dim"(%tensor, %index) {hello = "world"} : (tensor<?x?xf32>, index) -> index
// CHECK-GENERIC-NEXT:   %cast1 = "tensor.cast"(%tensor) : (tensor<?x?xf32>) -> tensor<4x4xf32>
// CHECK-GENERIC-NEXT:   %cast2 = "tensor.cast"(%tensor) {hello = "world"} : (tensor<?x?xf32>) -> tensor<4x4xf32>
// CHECK-GENERIC-NEXT:   %extract1 = "tensor.extract"(%tensor, %index, %index1) : (tensor<?x?xf32>, index, index) -> f32
// CHECK-GENERIC-NEXT:   %insert1 = "tensor.insert"(%extract1, %tensor, %index, %index1) : (f32, tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
// CHECK-GENERIC-NEXT:   %fromelements = "tensor.from_elements"(%index, %index1) : (index, index) -> tensor<2xindex>
// CHECK-GENERIC-NEXT:   %res_collapse1 = "tensor.collapse_shape"(%tensor) <{reassociation = [[0 : i64, 1 : i64], [2 : i64]]}> : (tensor<?x?xf32>) -> tensor<4x1xf32>
// CHECK-GENERIC-NEXT:   %expanded = "tensor.expand_shape"(%tensor, %dim1, %dim2) <{reassociation = [[0 : i64, 1 : i64, 2 : i64], [3 : i64]], static_output_shape = array<i64: -9223372036854775808, 1, 1, -9223372036854775808>}> : (tensor<?x?xf32>, index, index) -> tensor<?x1x1x?xf32>
// CHECK-GENERIC-NEXT:   %expanded_with_attr_dict = "tensor.expand_shape"(%tensor, %dim1, %dim2) <{reassociation = [[0 : i64, 1 : i64, 2 : i64], [3 : i64]], static_output_shape = array<i64: -9223372036854775808, 1, 1, -9223372036854775808>}> {test_attr = 42 : i8} : (tensor<?x?xf32>, index, index) -> tensor<?x1x1x?xf32>
// CHECK-GENERIC-NEXT:   %expanded_generic = "tensor.expand_shape"(%tensor, %dim1, %dim2) <{reassociation = [[0 : i64, 1 : i64, 2 : i64], [3 : i64]], static_output_shape = array<i64: -9223372036854775808, 1, 1, -9223372036854775808>}> : (tensor<?x?xf32>, index, index) -> tensor<?x1x1x?xf32>
// CHECK-GENERIC-NEXT:   %s = "test.op"() : () -> f32
// CHECK-GENERIC-NEXT:   %v = "tensor.splat"(%s) : (f32) -> tensor<8xf32>
// CHECK-GENERIC-NEXT:   %v2 = "tensor.splat"(%s, %index, %index1) : (f32, index, index) -> tensor<?x8x?xf32>
// CHECK-GENERIC-NEXT:   %pval, %ptensor = "test.op"() : () -> (f32, tensor<12x20x20xf32>)
// CHECK-GENERIC-NEXT:   %padded = "tensor.pad"(%ptensor, %dim1, %dim2) <{static_low = array<i64: 1, -9223372036854775808, 2>, static_high = array<i64: -9223372036854775808, 3, 3>, operandSegmentSizes = array<i32: 1, 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^bb0(%arg0 : index, %arg1 : index, %arg2 : index):
// CHECK-GENERIC-NEXT:     "tensor.yield"(%pval) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) : (tensor<12x20x20xf32>, index, index) -> tensor<?x?x25xf32>
// CHECK-GENERIC-NEXT: }) : () -> ()
