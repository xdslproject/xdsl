"builtin.module"() ({
  %0:3 = "test.op"() : () -> (tensor<5x3x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>)
  %1 = "onnx.Gemm"(%0#0, %0#1, %0#2) {onnx_node_name = "/Gemm"} : (tensor<5x3x3xf32>, tensor<3x2xf32>, tensor<5x2xf32>) -> tensor<5x2xf32>
}) : () -> ()
