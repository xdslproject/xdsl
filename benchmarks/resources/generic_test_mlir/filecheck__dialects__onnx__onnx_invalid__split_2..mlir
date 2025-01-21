"builtin.module"() ({
  %0:2 = "test.op"() : () -> (f32, tensor<2x4xf32>)
  %1 = "onnx.Mul"(%0#0, %0#1) {onnx_node_name = "/Mul"} : (f32, tensor<2x4xf32>) -> tensor<2x4xf32>
}) : () -> ()
