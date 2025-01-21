"builtin.module"() ({
  %0:3 = "test.op"() : () -> (tensor<2x4xf32>, tensor<3x2xf32>, tensor<3x2xf32>)
  %1 = "onnx.Gemm"(%0#0, %0#1, %0#2) {onnx_node_name = "/Gemm"} : (tensor<2x4xf32>, tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<2x4xf32>
}) : () -> ()
