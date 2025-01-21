"builtin.module"() ({
  %0:2 = "test.op"() : () -> (f32, tensor<2x4xi64>)
  %1 = "onnx.Reshape"(%0#0, %0#1) {onnx_node_name = "/Reshape"} : (f32, tensor<2x4xi64>) -> tensor<2x4xi64>
}) : () -> ()
