"builtin.module"() ({
  %0:2 = "test.op"() : () -> (tensor<4x3x2xf32>, tensor<1xi64>)
  %1 = "onnx.Reshape"(%0#0, %0#1) {onnx_node_name = "/Reshape"} : (tensor<4x3x2xf32>, tensor<1xi64>) -> tensor<4x3x2xi32>
}) : () -> ()
