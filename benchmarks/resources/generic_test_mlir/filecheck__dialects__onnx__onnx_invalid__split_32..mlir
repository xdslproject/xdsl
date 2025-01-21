"builtin.module"() ({
  %0:3 = "test.op"() : () -> (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none)
  %1 = "onnx.Conv"(%0#0, %0#1, %0#2) {auto_pad = "INVALID", dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], onnx_node_name = "/Conv", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>
}) : () -> ()
