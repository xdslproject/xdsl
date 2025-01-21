"builtin.module"() ({
  %0:2 = "test.op"() : () -> (tensor<2x4xf32>, tensor<1x4xf32>)
  %1 = "onnx.Sub"(%0#0, %0#1) {onnx_node_name = "/Sub"} : (tensor<2x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
}) : () -> ()
