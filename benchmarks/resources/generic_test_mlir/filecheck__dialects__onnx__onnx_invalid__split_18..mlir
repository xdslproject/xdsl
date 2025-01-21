"builtin.module"() ({
  %0:2 = "test.op"() : () -> (tensor<5x5xf32>, tensor<2x2xi64>)
  %1 = "onnx.Reshape"(%0#0, %0#1) {onnx_node_name = "/Reshape"} : (tensor<5x5xf32>, tensor<2x2xi64>) -> tensor<5x5xf32>
}) : () -> ()
