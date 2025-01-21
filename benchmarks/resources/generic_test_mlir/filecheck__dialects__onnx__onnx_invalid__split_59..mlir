"builtin.module"() ({
  %0 = "test.op"() : () -> tensor<3x4xf32>
  %1 = "onnx.Sigmoid"(%0) {onnx_node_name = "/Sigmoid"} : (tensor<3x4xf32>) -> tensor<7x3xf32>
}) : () -> ()
