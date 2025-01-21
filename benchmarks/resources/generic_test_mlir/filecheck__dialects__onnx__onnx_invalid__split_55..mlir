"builtin.module"() ({
  %0 = "test.op"() : () -> tensor<3x4xf32>
  %1 = "onnx.Transpose"(%0) {onnx_node_name = "/Transpose", perm = [1, 2]} : (tensor<3x4xf32>) -> tensor<4x3xf32>
}) : () -> ()
