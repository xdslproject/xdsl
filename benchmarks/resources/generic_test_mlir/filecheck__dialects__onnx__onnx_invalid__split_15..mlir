"builtin.module"() ({
  %0:2 = "test.op"() : () -> (tensor<6x9x5xf32>, tensor<3xi32>)
  %1 = "onnx.Reshape"(%0#0, %0#1) {onnx_node_name = "/Reshape"} : (tensor<6x9x5xf32>, tensor<3xi32>) -> tensor<6x9xf32>
}) : () -> ()
