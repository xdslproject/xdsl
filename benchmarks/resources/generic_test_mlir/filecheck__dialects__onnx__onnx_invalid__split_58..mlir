"builtin.module"() ({
  %0 = "test.op"() : () -> tensor<1x2x4xf32>
  %1 = "onnx.Squeeze"(%0) {axes = 3 : i64, onnx_node_name = "/Squeeze"} : (tensor<1x2x4xf32>) -> tensor<2x4xf32>
}) : () -> ()
