"builtin.module"() ({
  %0 = "test.op"() : () -> tensor<5x5x32x32xf32>
  %1 = "onnx.MaxPoolSingleOut"(%0) {auto_pad = "NOTSET", ceil_mode = 0 : i64, dilations = [1, 1], kernel_shape = [3, 3], onnx_node_name = "/MaxPoolSingleOut", pads = [0, 0, 0, 0], storage_order = 1 : i64, strides = [1, 1]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>
}) : () -> ()
