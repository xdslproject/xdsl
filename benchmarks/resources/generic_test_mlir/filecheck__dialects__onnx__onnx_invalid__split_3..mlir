"builtin.module"() ({
  %0:2 = "test.op"() : () -> (tensor<2x4xf32>, tensor<2x4xi32>)
  %1 = "onnx.Div"(%0#0, %0#1) {onnx_node_name = "/Div"} : (tensor<2x4xf32>, tensor<2x4xi32>) -> tensor<2x4xf32>
}) : () -> ()
