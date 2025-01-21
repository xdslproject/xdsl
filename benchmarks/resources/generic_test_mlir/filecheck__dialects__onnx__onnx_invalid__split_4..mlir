"builtin.module"() ({
  %0 = "test.op"() : () -> tensor<2x4xf32>
  %1 = "onnx.Relu"(%0) {onnx_node_name = "/Relu"} : (tensor<2x4xf32>) -> tensor<3x4xf32>
}) : () -> ()
