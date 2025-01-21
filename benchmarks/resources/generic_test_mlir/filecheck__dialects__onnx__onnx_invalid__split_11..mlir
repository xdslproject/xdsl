"builtin.module"() ({
  %0:3 = "test.op"() : () -> (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2x7xf32>)
  %1 = "onnx.Gemm"(%0#0, %0#1, %0#2) {beta = 4.700000e+01 : f32, onnx_node_name = "/Gemm"} : (tensor<5x3xf32>, tensor<3x2xf32>, tensor<5x2x7xf32>) -> tensor<5x3xf32>
}) : () -> ()
