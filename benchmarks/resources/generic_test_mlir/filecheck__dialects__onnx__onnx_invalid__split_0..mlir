"builtin.module"() ({
  %0:2 = "test.op"() : () -> (tensor<2x4xf32>, tensor<3x2xf32>)
  %1 = "onnx.Add"(%0#0, %0#1) {onnx_node_name = "/Add"} : (tensor<2x4xf32>, tensor<3x2xf32>) -> tensor<2x4xf32>
}) : () -> ()
