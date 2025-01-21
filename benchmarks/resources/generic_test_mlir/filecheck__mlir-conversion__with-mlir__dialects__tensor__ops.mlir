"builtin.module"() ({
  %0 = "tensor.empty"() : () -> tensor<2x3xf32>
  %1 = "tensor.empty"() : () -> tensor<2xf32>
  %2 = "test.op"() : () -> index
  %3 = "tensor.empty"(%2) : (index) -> tensor<?xf32>
  %4 = "test.op"() {value = dense<1.000000e-01> : tensor<4x1xf32>} : () -> tensor<4x1xf32>
  %5 = "test.op"() : () -> tensor<1xi32>
  %6 = "tensor.reshape"(%4, %5) : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>
  %7 = "tensor.insert_slice"(%1, %0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 1>, static_sizes = array<i64: 1, 2>, static_strides = array<i64: 1, 1>}> : (tensor<2xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %8 = "tensor.extract_slice"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 1>, static_sizes = array<i64: 1, 2>, static_strides = array<i64: 1, 1>}> : (tensor<2x3xf32>) -> tensor<2xf32>
  %9 = "tensor.dim"(%0, %2) : (tensor<2x3xf32>, index) -> index
  %10 = "tensor.dim"(%0, %2) {hello = "world"} : (tensor<2x3xf32>, index) -> index
  %11 = "tensor.cast"(%0) : (tensor<2x3xf32>) -> tensor<?x?xf32>
  %12 = "tensor.cast"(%0) {hello = "world"} : (tensor<2x3xf32>) -> tensor<?x?xf32>
  %13 = "tensor.empty"() : () -> tensor<2x3x2x3xf32>
  %14 = "tensor.collapse_shape"(%13) <{reassociation = [[0, 1], [2, 3]]}> : (tensor<2x3x2x3xf32>) -> tensor<6x6xf32>
  %15 = "tensor.extract"(%1, %2) : (tensor<2xf32>, index) -> f32
  %16 = "tensor.insert"(%15, %1, %2) : (f32, tensor<2xf32>, index) -> tensor<2xf32>
}) : () -> ()
