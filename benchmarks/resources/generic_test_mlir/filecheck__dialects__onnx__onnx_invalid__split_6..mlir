"builtin.module"() ({
  %0:3 = "test.op"() : () -> (f32, tensor<2x4xf32>, tensor<2x4xf32>)
  %1 = "onnx.Gemm"(%0#0, %0#1, %0#2) {onnx_node_name = "/Gemm"} : (f32, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
}) : () -> ()
