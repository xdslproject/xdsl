"builtin.module"() ({
  %0:2 = "test.op"() : () -> (tensor<3x3xf32>, tensor<?xi64>)
  %1 = "onnx.Reshape"(%0#0, %0#1) {onnx_node_name = "/Reshape"} : (tensor<3x3xf32>, tensor<?xi64>) -> tensor<3x3xf32>
}) : () -> ()
