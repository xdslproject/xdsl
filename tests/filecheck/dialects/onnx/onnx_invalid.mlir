// RUN: xdsl-opt --verify-diagnostics --split-input-file %s | filecheck %s

// Non-broadcastable operands are not allowed.

builtin.module {
  %t0, %t1 = "test.op"() : () -> (tensor<2x4xf32>, tensor<3x2xf32>)

  // CHECK: Operation does not verify: operands have incompatible shapes: (2, 4) and (3, 2)
  %res_add =  "onnx.Add"(%t0, %t1) {onnx_node_name = "/Add"} : (tensor<2x4xf32>, tensor<3x2xf32>) -> tensor<2x4xf32>
}

// -----

builtin.module {
  %t0, %t1 = "test.op"() : () -> (tensor<2x4xf32>, tensor<1x4xf32>)

  // CHECK: Operation does not verify: result shape [2, 4] does not match result type tensor<1x4xf32>
  %res_sub =  "onnx.Sub"(%t0, %t1) {onnx_node_name = "/Sub"} : (tensor<2x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
}

// -----

builtin.module {
  %t0, %t1 = "test.op"() : () -> (f32, tensor<2x4xf32>)

  // CHECK: operand at position 0 does not verify:
  // CHECK: f32 should be of base attribute tensor
  %res_mul =  "onnx.Mul"(%t0, %t1) {onnx_node_name = "/Mul"} : (f32, tensor<2x4xf32>) -> tensor<2x4xf32>
}

// -----

builtin.module {
  %t0, %t1 = "test.op"() : () -> (tensor<2x4xf32>, tensor<2x4xi32>)

  // CHECK: operand at position 1 does not verify:
  // CHECK: attribute f32 expected from variable 'T', but got i32
  %res_div =  "onnx.Div"(%t0, %t1) {onnx_node_name = "/Div"} : (tensor<2x4xf32>, tensor<2x4xi32>) -> tensor<2x4xf32>
}

// -----

builtin.module {
  %t0 = "test.op"() : () -> (tensor<2x4xf32>)

  // CHECK: operand at position 1 does not verify:
  // CHECK: Operation does not verify: Mismatch between operand type and res type of onnx.Relu
  %res_relu =  "onnx.Relu"(%t0) {onnx_node_name = "/Relu"} : (tensor<2x4xf32>) -> tensor<3x4xf32>
}

// -----

builtin.module {
  %t0, %t1, %t2 = "test.op"() : () -> (tensor<2x4xf32>, tensor<3x2xf32>, tensor<3x2xf32>)

  // CHECK: Operation does not verify: operands have incompatible shapes: (2, 4) and (3, 2)
  %res_gemm =  "onnx.Gemm"(%t0, %t1, %t2) {onnx_node_name = "/Gemm"} : (tensor<2x4xf32>, tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<2x4xf32>
}

// -----

builtin.module {
  %t0, %t1, %t2 = "test.op"() : () -> (f32, tensor<2x4xf32>,tensor<2x4xf32>)

  // CHECK: operand at position 0 does not verify:
  // CHECK: f32 should be of base attribute tensor
  %res_gemm=  "onnx.Gemm"(%t0, %t1, %t2) {onnx_node_name = "/Gemm"} : (f32, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
}

// -----

builtin.module {
    %t0, %t1, %t2 = "test.op"(): () -> (tensor<5x2xf32>, tensor<2x1xf32>, tensor<5x4xf32>)

    //CHECK:  Operation does not verify: result shape [5, 4] does not match result type tensor<5x2xf32>
    %res_gemm = "onnx.Gemm"(%t0, %t1, %t2) : (tensor<5x2xf32>, tensor<2x1xf32>, tensor<5x4xf32>) -> tensor<5x2xf32>
}

// -----

builtin.module {
  %t0, %t1, %t2 = "test.op"() : () -> (tensor<2x4xf32>, tensor<2x4xi32>, tensor<2x4xf32>)

  // CHECK: operand at position 1 does not verify:
  // CHECK: attribute f32 expected from variable 'T', but got i32
  %res_gemm =  "onnx.Gemm"(%t0, %t1, %t2) {onnx_node_name = "/Gemm"} : (tensor<2x4xf32>, tensor<2x4xi32>, tensor<2x4xf32>) -> tensor<2x4xf32>

 }

 // -----

builtin.module {
  %t0, %t1, %t2 = "test.op"() : () -> (tensor<5x3x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>)
  // CHECK: Operation does not verify: tensor A should be a 2D tensor
  %res_gemm = "onnx.Gemm"(%t0, %t1, %t2) {onnx_node_name = "/Gemm"}: (tensor<5x3x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>) -> tensor<5x2xf32>
}

// -----

builtin.module {
  %t0, %t1, %t2 = "test.op"() : () -> (tensor<5x3xf32>, tensor<3x2x3xf32>, tensor<5x2xf32>)
  // CHECK: Operation does not verify: tensor B should be a 2D tensor
  %res_gemm = "onnx.Gemm"(%t0, %t1, %t2) {onnx_node_name = "/Gemm"}: (tensor<5x3xf32>, tensor<3x2x3xf32>, tensor<5x2xf32>) -> tensor<5x2xf32>
}

// -----

builtin.module {
  %t0, %t1, %t2 = "test.op"() : () -> (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2x7xf32>)
  // CHECK: Operation does not verify: tensor C should be a 1D tensor or 2D tensor
  %res_gemm = "onnx.Gemm"(%t0, %t1, %t2) {onnx_node_name = "/Gemm",  beta = 47.0 : f32}: (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2x7xf32>) -> tensor<5x3xf32>
}

// -----

builtin.module {
  %t0, %t1 = "test.op"() : () -> (f32, tensor<2x4xi64>)

  // CHECK: operand at position 0 does not verify:
  // CHECK: f32 should be of base attribute tensor
  %res_reshape =  "onnx.Reshape"(%t0, %t1) {onnx_node_name = "/Reshape"} : (f32, tensor<2x4xi64>) -> tensor<2x4xi64>
}

// -----

builtin.module {
    %t0, %t1 = "test.op"() : () -> (tensor<4x3x2xf32>, tensor<1xi64>)

  // CHECK: result at position 0 does not verify:
  // CHECK: attribute f32 expected from variable 'T', but got i32
  %res_reshape = "onnx.Reshape"(%t0, %t1) {"onnx_node_name" = "/Reshape"} : (tensor<4x3x2xf32>, tensor<1xi64>) -> tensor<4x3x2xi32>
}

// -----

builtin.module {
    %t0, %t1 = "test.op"() : () -> (tensor<6x9x5xf32>, tensor<3xi64>)

  // CHECK: result at position 0 does not verify:
  // CHECK: Operation does not verify: Input tensor's shape and output tensor's shape must have the same number of elements
  %res_reshape = "onnx.Reshape"(%t0, %t1) {"onnx_node_name" = "/Reshape"} : (tensor<6x9x5xf32>, tensor<3xi64>) -> tensor<6x9xf32>
}

// -----

builtin.module {
    %t0, %t1 = "test.op"() : () -> (tensor<6x9x5xf32>, tensor<3xi32>)

  // CHECK: Operation does not verify: shape element type has to be a 64-bit signless integer
  %res_reshape = "onnx.Reshape"(%t0, %t1) {"onnx_node_name" = "/Reshape"} : (tensor<6x9x5xf32>, tensor<3xi32>) -> tensor<6x9xf32>
}

// -----

builtin.module {
    %t0, %t1 = "test.op"() : () -> (tensor<6x9x5xf32>, tensor<3xi64>)

  // CHECK: result at position 0 does not verify:
  // CHECK:  vector<6x9x5xf32> should be of base attribute tensor
  %res_reshape = "onnx.Reshape"(%t0, %t1) {"onnx_node_name" = "/Reshape"} : (tensor<6x9x5xf32>, tensor<3xi64>) -> vector<6x9x5xf32>
}

// -----

builtin.module {
    %t0, %t1 = "test.op"() : () -> (vector<6x9x5xf32>, tensor<3xi64>)

  // CHECK: operand at position 0 does not verify:
  // CHECK:  vector<6x9x5xf32> should be of base attribute tensor
  %res_reshape = "onnx.Reshape"(%t0, %t1) {"onnx_node_name" = "/Reshape"} : (vector<6x9x5xf32>, tensor<3xi64>) -> tensor<6x9x5xf32>
}

// -----

builtin.module {
    %t0, %t1 = "test.op"() : () -> (tensor<5x5xf32>, tensor<2x2xi64>)

    //CHECK: Operation does not verify: Shape tensor must have a rank one
    %res_reshape = "onnx.Reshape"(%t0, %t1) {onnx_node_name = "/Reshape"}: (tensor<5x5xf32>, tensor<2x2xi64>) -> tensor<5x5xf32>

}

// -----

builtin.module {
    %t0, %t1 = "test.op"() : () -> (tensor<3x3xf32>, tensor<?xi64>)

    //CHECK: Operation does not verify: Shape tensor rank must not be equal to -1
    %res_reshape = "onnx.Reshape"(%t0, %t1) {onnx_node_name = "/Reshape"}: (tensor<3x3xf32>, tensor<?xi64>) -> tensor<3x3xf32>

}

// -----

builtin.module {
    %t0 = "test.op"() : () -> (tensor<3x3xf32>)

    //CHECK: Operation does not verify: Mismatch between operand type and res type of onnx.Abs
    %res_abs = "onnx.Abs"(%t0) {onnx_node_name = "/Abs"}: (tensor<3x3xf32>) -> tensor<2x3xf32>

}


// -----

builtin.module {
  %t0,%t1,%t2 = "test.op"(): () ->  (f32, tensor<1x1x3x3xf32>, none)

  // CHECK: operand at position 0 does not verify:
  // CHECK: f32 should be of base attribute tensor
  %res_conv =  "onnx.Conv"(%t0, %t1, %t2) {onnx_node_name = "/Conv"} : (f32, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>
}

// -----

builtin.module {
    %t0,%t1,%t2 = "test.op"(): () ->  (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none)

  // CHECK: result at position 0 does not verify:
  // CHECK: attribute f32 expected from variable 'T', but got i32
  %res_conv = "onnx.Conv"(%t0, %t1, %t2) {"onnx_node_name" = "/Conv"} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xi32>

}

// -----

builtin.module {
    %t0,%t1,%t2 = "test.op"(): () ->  (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, tensor<4x2xf32>)

  // CHECK: Operation does not verify: bias must be 1D
  %res_conv = "onnx.Conv"(%t0, %t1, %t2) {"onnx_node_name" = "/Conv", "auto_pad" = "NOTSET", "dilations" = [1 : i64, 1: i64], "group" = 1 : i64, "kernel_shape" = [4 : i64, 4 : i64], "pads" = [0 : i64, 0: i64, 0: i64, 0: i64], "strides" = [1: i64, 1: i64]} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, tensor<4x2xf32>) -> tensor<1x1x3x3xf32>

}

// -----

builtin.module {
    %t0,%t1,%t2 = "test.op"(): () ->  (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none)

  // CHECK: Operation does not verify: kernel shape rank and weight tensor rank are not the same
  %res_conv = "onnx.Conv"(%t0, %t1, %t2) {"onnx_node_name" = "/Conv", "auto_pad" = "NOTSET", "dilations" = [1 : i64, 1: i64], "group" = 1 : i64, "kernel_shape" = [4 : i64, 4 : i64], "pads" = [0 : i64, 0: i64, 0: i64, 0: i64], "strides" = [1: i64, 1: i64]} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>

}

// -----

builtin.module {
    %t0,%t1,%t2 = "test.op"(): () ->  (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none)

  // CHECK: Operation does not verify: dilation value must be non zero positive
  %res_conv = "onnx.Conv"(%t0, %t1, %t2) {"onnx_node_name" = "/Conv", "auto_pad" = "NOTSET", "dilations" = [-2 : i64, -2: i64], "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "pads" = [0 : i64, 0: i64, 0: i64, 0: i64], "strides" = [1: i64, 1: i64]} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>
}

// -----

builtin.module {
    %t0,%t1,%t2 = "test.op"(): () ->  (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none)

  // CHECK: Operation does not verify: dilations rank and kernel shape rank are not the same
  %res_conv = "onnx.Conv"(%t0, %t1, %t2) {"onnx_node_name" = "/Conv", "auto_pad" = "NOTSET", "dilations" = [1 : i64, 1: i64, 3: i64], "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "pads" = [0 : i64, 0: i64, 0: i64, 0: i64], "strides" = [1: i64, 1: i64]} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>
}

// -----

builtin.module {
    %t0,%t1,%t2 = "test.op"(): () ->  (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none)

  // CHECK: Operation does not verify: group value must be nonnegative
  %res_conv = "onnx.Conv"(%t0, %t1, %t2) {"onnx_node_name" = "/Conv", "auto_pad" = "NOTSET", "dilations" = [1 : i64, 1: i64], "group" = 0 : i64, "kernel_shape" = [3 : i64, 3 : i64], "pads" = [0 : i64, 0: i64, 0: i64, 0: i64], "strides" = [1: i64, 1: i64]} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>
}


// -----

builtin.module {
    %t0,%t1,%t2 = "test.op"(): () ->  (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none)

  // CHECK: Operation does not verify: stride value must be non zero positive
  %res_conv = "onnx.Conv"(%t0, %t1, %t2) {"onnx_node_name" = "/Conv", "auto_pad" = "NOTSET", "dilations" = [1 : i64, 1: i64], "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "pads" = [0 : i64, 0: i64, 0: i64, 0: i64], "strides" = [-2 : i64, -2: i64]} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>
}

// -----

builtin.module {
    %t0,%t1,%t2 = "test.op"(): () ->  (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none)

  // CHECK: Operation does not verify: strides rank and kernel shape rank are not the same
  %res_conv = "onnx.Conv"(%t0, %t1, %t2) {"onnx_node_name" = "/Conv", "auto_pad" = "NOTSET", "dilations" = [1 : i64, 1: i64], "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "pads" = [0 : i64, 0: i64, 0: i64, 0: i64], "strides" = [1: i64, 1: i64, 1: i64]} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>
}

// -----

builtin.module {
    %t0,%t1,%t2 = "test.op"(): () ->  (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none)

  // CHECK: Operation does not verify: pads value must be nonnegative
  %res_conv = "onnx.Conv"(%t0, %t1, %t2) {"onnx_node_name" = "/Conv", "auto_pad" = "NOTSET", "dilations" = [1 : i64, 1: i64], "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "pads" = [-1 : i64, -1: i64, -1: i64, -1: i64], "strides" = [1: i64, 1: i64]} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>
}


// -----

builtin.module {
    %t0,%t1,%t2 = "test.op"(): () ->  (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none)

  // CHECK: Operation does not verify: pads rank is not twice the kernel shape rank
  %res_conv = "onnx.Conv"(%t0, %t1, %t2) {"onnx_node_name" = "/Conv", "auto_pad" = "NOTSET", "dilations" = [1 : i64, 1: i64], "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "pads" = [1 : i64, 1: i64, 1: i64], "strides" = [1: i64, 1: i64]} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>
}

// -----

builtin.module {
    %t0,%t1,%t2 = "test.op"(): () ->  (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none)

  // CHECK: Operation does not verify:  Invalid auto_pad string. Must be one of ['NOTSET', 'SAME_UPPER', 'SAME_LOWER', 'VALID']
  %res_conv = "onnx.Conv"(%t0, %t1, %t2) {"onnx_node_name" = "/Conv", "auto_pad" = "INVALID", "dilations" = [1 : i64, 1: i64], "group" = 1 : i64, "kernel_shape" = [3 : i64, 3 : i64], "pads" = [0 : i64, 0: i64, 0: i64, 0: i64], "strides" = [1: i64, 1: i64]} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>

  }


  // -----

builtin.module {

  // CHECK: f32 should be of base attribute tensor
  %res_constant = "onnx.Constant"() {value = dense<[3.0]> : tensor<1xf32>} : () -> f32

  }

// -----

builtin.module {

  // CHECK: Operation does not verify: value attribute type must be of type TensorType
  %res_constant = "onnx.Constant"() {value = dense<[3.0]> : vector<1xf32>} : () -> tensor<1xf32>

  }

// -----

builtin.module {

  // CHECK: Operation does not verify: value_int element type has to be a 64-bit signless integer
  %res_constant = "onnx.Constant"() {value_int = 4 : i32} : () -> tensor<1xf32>

  }

// -----

builtin.module {

  // CHECK: Operation does not verify: value_ints elements type has to be a 64-bit signless integer
  %res_constant = "onnx.Constant"() {value_ints = [1: i64, 2: i32, 3: i64]} : () -> tensor<3xi32>

  }

// -----

builtin.module {

  // CHECK: Operation does not verify: Only one value attribute must be provided, but 2 were specified
  %res_constant = "onnx.Constant"() {value_ints = [1: i64, 1: i64], value_int =  3: i64} : () -> tensor<3xi64>

  }

// -----

builtin.module {
  %t0 = "test.op"(): () ->  (f32)

  // CHECK: operand at position 0 does not verify:
  // CHECK: Expected tensor or memref type, got f32
  %res_max_pool_single_out =  "onnx.MaxPoolSingleOut"(%t0) {onnx_node_name = "/MaxPoolSingleOut"} : (f32) -> tensor<5x5x32x32xf32>
}

// -----

builtin.module {
    %t0= "test.op"(): () ->  (tensor<5x5x32x32xf32>)

  // CHECK: result at position 0 does not verify:
  // CHECK: attribute f32 expected from variable 'T', but got i32
  %res_max_pool_single_out = "onnx.MaxPoolSingleOut"(%t0) {"onnx_node_name" = "/MaxPoolSingleOut"} : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xi32>

}

// -----

builtin.module {
    %t0 = "test.op"(): () ->  (tensor<5x5x32x32xf32>)

  // CHECK: Operation does not verify:  Invalid auto_pad string. Must be one of ['NOTSET', 'SAME_UPPER', 'SAME_LOWER', 'VALID']
  %res_max_pool_single_out = "onnx.MaxPoolSingleOut"(%t0) {"onnx_node_name" = "/MaxPoolSingleOut", "auto_pad"  = "INVALID", "ceil_mode" = 0 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "pads" = [0 : i64, 0 : i64, 0 : i64, 0 : i64], "storage_order" = 0 : i64, "strides" = [1 : i64, 1 : i64]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>

}

// -----

builtin.module {
    %t0 = "test.op"(): () ->  (tensor<5x5x32x32xf32>)

  // CHECK: Operation does not verify:  ceil value must be either zero or one
  %res_max_pool_single_out = "onnx.MaxPoolSingleOut"(%t0) {"onnx_node_name" = "/MaxPoolSingleOut", "auto_pad"  = "NOTSET", "ceil_mode" = 2 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "pads" = [0 : i64, 0 : i64, 0 : i64, 0 : i64], "storage_order" = 0 : i64, "strides" = [1 : i64, 1 : i64]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>

}

// -----

builtin.module {
    %t0 = "test.op"(): () ->  (tensor<5x5x32x32xf32>)

  // CHECK: Operation does not verify: input data and kernel shape rank mismatch: (2) vs (1)
  %res_max_pool_single_out = "onnx.MaxPoolSingleOut"(%t0) {"onnx_node_name" = "/MaxPoolSingleOut", "auto_pad"  = "NOTSET", "ceil_mode" = 0 : i64, "kernel_shape" = [3 : i64], "dilations" = [1 : i64, 1 : i64], "pads" = [0 : i64, 0 : i64, 0 : i64, 0 : i64], "storage_order" = 0 : i64, "strides" = [1 : i64, 1 : i64]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>

}

// -----

builtin.module {
    %t0 = "test.op"(): () ->  (tensor<5x5x32x32xf32>)

  // CHECK: Operation does not verify: dilation value must be non zero positive
  %res_max_pool_single_out = "onnx.MaxPoolSingleOut"(%t0) {"onnx_node_name" = "/MaxPoolSingleOut", "auto_pad"  = "NOTSET", "ceil_mode" = 0 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [-1 : i64, -1 : i64], "pads" = [0 : i64, 0 : i64, 0 : i64, 0 : i64], "storage_order" = 0 : i64, "strides" = [1 : i64, 1 : i64]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>

}

// -----

builtin.module {
    %t0 = "test.op"(): () ->  (tensor<5x5x32x32xf32>)

  // CHECK: Operation does not verify: dilations rank (3) and kernel shape rank (2) are not the same
  %res_max_pool_single_out = "onnx.MaxPoolSingleOut"(%t0) {"onnx_node_name" = "/MaxPoolSingleOut", "auto_pad"  = "NOTSET", "ceil_mode" = 0 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64, 1: i64], "pads" = [0 : i64, 0 : i64, 0 : i64, 0 : i64], "storage_order" = 0 : i64, "strides" = [1 : i64, 1 : i64]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>

}

// -----

builtin.module {
    %t0 = "test.op"(): () ->  (tensor<5x5x32x32xf32>)

  // CHECK: Operation does not verify: column major storage order not implemented yet
  %res_max_pool_single_out = "onnx.MaxPoolSingleOut"(%t0) {"onnx_node_name" = "/MaxPoolSingleOut", "auto_pad"  = "NOTSET", "ceil_mode" = 0 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "pads" = [0 : i64, 0 : i64, 0 : i64, 0 : i64], "storage_order" = 1 : i64, "strides" = [1 : i64, 1 : i64]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>

}

// -----

builtin.module {
    %t0 = "test.op"(): () ->  (tensor<5x5x32x32xf32>)

  // CHECK: Operation does not verify: stride value must be non zero positive
  %res_max_pool_single_out = "onnx.MaxPoolSingleOut"(%t0) {"onnx_node_name" = "/MaxPoolSingleOut", "auto_pad"  = "NOTSET", "ceil_mode" = 0 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "pads" = [0 : i64, 0 : i64, 0 : i64, 0 : i64], "storage_order" = 0 : i64, "strides" = [-1 : i64, -1 : i64]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>

}

// -----

builtin.module {
    %t0 = "test.op"(): () ->  (tensor<5x5x32x32xf32>)

  // CHECK: Operation does not verify: strides rank (3) and kernel shape rank (2) are not the same
  %res_max_pool_single_out = "onnx.MaxPoolSingleOut"(%t0) {"onnx_node_name" = "/MaxPoolSingleOut", "auto_pad"  = "NOTSET", "ceil_mode" = 0 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "pads" = [0 : i64, 0 : i64, 0 : i64, 0 : i64], "storage_order" = 0 : i64, "strides" = [1 : i64, 1 : i64, 1: i64]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>

}

// -----

builtin.module {
    %t0 = "test.op"(): () ->  (tensor<5x5x32x32xf32>)

  // CHECK: Operation does not verify: pads value must be nonnegative
  %res_max_pool_single_out = "onnx.MaxPoolSingleOut"(%t0) {"onnx_node_name" = "/MaxPoolSingleOut", "auto_pad"  = "NOTSET", "ceil_mode" = 0 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "pads" = [-2 : i64, 0 : i64, 0 : i64, 0 : i64], "storage_order" = 0 : i64, "strides" = [1 : i64, 1 : i64]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>

}

// -----

builtin.module {
    %t0 = "test.op"(): () ->  (tensor<5x5x32x32xf32>)

  // CHECK: Operation does not verify: pads rank (5) is not twice the kernel shape rank (2)
  %res_max_pool_single_out = "onnx.MaxPoolSingleOut"(%t0) {"onnx_node_name" = "/MaxPoolSingleOut", "auto_pad"  = "NOTSET", "ceil_mode" = 0 : i64, "kernel_shape" = [3 : i64, 3 : i64], "dilations" = [1 : i64, 1 : i64], "pads" = [0 : i64, 0 : i64, 0 : i64, 0 : i64, 0: i64], "storage_order" = 0 : i64, "strides" = [1 : i64, 1 : i64]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>

}

// -----

builtin.module {
  %t0, %t1 = "test.op"() : () -> (tensor<2x4x3xf32>, tensor<4x2xf32>)

  // CHECK: Operation does not verify: input matrix A should be a 2D tensor
  %res_matmul =  "onnx.MatMul"(%t0, %t1) {onnx_node_name = "/MatMul"} : (tensor<2x4x3xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
}

// -----

builtin.module {
  %t0, %t1 = "test.op"() : () -> (tensor<2x4xf32>, tensor<4x2x3xf32>)

  // CHECK: Operation does not verify: input matrix B should be a 2D tensor
  %res_matmul =  "onnx.MatMul"(%t0, %t1) {onnx_node_name = "/MatMul"} : (tensor<2x4xf32>, tensor<4x2x3xf32>) -> tensor<2x2xf32>
}

// -----

builtin.module {
  %t0, %t1 = "test.op"() : () -> (tensor<2x4xf32>, tensor<5x2xf32>)

  // CHECK: Operation does not verify: operands have incompatible shapes: (2, 4) and (5, 2)
  %res_matmul =  "onnx.MatMul"(%t0, %t1) {onnx_node_name = "/MatMul"} : (tensor<2x4xf32>, tensor<5x2xf32>) -> tensor<2x2xf32>
}

// -----

builtin.module {
  %t0, %t1 = "test.op"() : () -> (tensor<2x4xf32>, tensor<4x2xf32>)

  // CHECK: Operation does not verify: result shape [2, 2] does not match result type [2, 3]
  %res_matmul =  "onnx.MatMul"(%t0, %t1) {onnx_node_name = "/MatMul"} : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x3xf32>
}

// -----

builtin.module {
  %t0 = "test.op"() : () -> (tensor<3x4xf32>)
  // CHECK: Operation does not verify: permutation can not contain more than one occurrence of the same dimension: dimension #1 appears 2 times.
  %res_transpose = "onnx.Transpose"(%t0) {onnx_node_name = "/Transpose", "perm" = [1 : i64, 1 : i64]}: (tensor<3x4xf32>) -> tensor<4x3xf32>
}

// -----

builtin.module {
  %t0 = "test.op"() : () -> (tensor<3x4xf32>)
  // CHECK: Operation does not verify: permutation can only contain values between 0 and 2-1: dimension #1 value is 2
  %res_transpose = "onnx.Transpose"(%t0) {onnx_node_name = "/Transpose", "perm" = [1 : i64, 2 : i64]}: (tensor<3x4xf32>) -> tensor<4x3xf32>
}

// -----

builtin.module {
  %t0 = "test.op"() : () -> (tensor<1x3x4xf32>)
  // CHECK: Operation does not verify: permutation and inputs dimensions must have the same size: #dimensions input is 3, #dimension perimutation is 2
  %res_transpose = "onnx.Transpose"(%t0) {onnx_node_name = "/Transpose", "perm" = [1 : i64, 0 : i64]}: (tensor<1x3x4xf32>) -> tensor<3x1x4xf32>
}

// -----

builtin.module {
  %t0 = "test.op"() : () -> (tensor<3x4xf32>)
  // CHECK: Operation does not verify: incorrect output shape: output dimension #0 should be equal to 4
  %res_transpose = "onnx.Transpose"(%t0) {onnx_node_name = "/Transpose", "perm" = [1 : i64, 0 : i64]}: (tensor<3x4xf32>) -> tensor<3x3xf32>
}

// -----

builtin.module {
  %t0 = "test.op"() : () -> (tensor<1x2x4xf32>)

  // CHECK: Operation does not verify: axes to squeeze must be between 0 and 2, axes: 3
  %res_squeeze =  "onnx.Squeeze"(%t0) {onnx_node_name = "/Squeeze", "axes" = 3 : i64} : (tensor<1x2x4xf32>) -> tensor<2x4xf32>
}

// -----

builtin.module {
  %t0 = "test.op"() : () -> (tensor<3x4xf32>)

  // CHECK: Operation does not verify: tensor input shape (3, 4) is not equal to tensor output shape (7, 3)
  %res_sigmoid =  "onnx.Sigmoid"(%t0) {onnx_node_name = "/Sigmoid"} : (tensor<3x4xf32>) -> tensor<7x3xf32>
}
