"builtin.module"() ({
  %0 = "test.op"() : () -> tensor<3x3xf32>
  %1 = "onnx.Abs"(%0) {onnx_node_name = "/Abs"} : (tensor<3x3xf32>) -> tensor<2x3xf32>
}) : () -> ()
