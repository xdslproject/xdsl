"builtin.module"() ({
  %0:2 = "test.op"() : () -> (vector<6x9x5xf32>, tensor<3xi64>)
  %1 = "onnx.Reshape"(%0#0, %0#1) {onnx_node_name = "/Reshape"} : (vector<6x9x5xf32>, tensor<3xi64>) -> tensor<6x9x5xf32>
}) : () -> ()
