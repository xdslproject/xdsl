"builtin.module"() ({
  %0 = "test.op"() : () -> tensor<1x3x4xf32>
  %1 = "onnx.Transpose"(%0) {onnx_node_name = "/Transpose", perm = [1, 0]} : (tensor<1x3x4xf32>) -> tensor<3x1x4xf32>
}) : () -> ()
