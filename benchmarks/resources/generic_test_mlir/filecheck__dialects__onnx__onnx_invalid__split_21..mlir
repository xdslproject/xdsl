"builtin.module"() ({
  %0:3 = "test.op"() : () -> (f32, tensor<1x1x3x3xf32>, none)
  %1 = "onnx.Conv"(%0#0, %0#1, %0#2) {onnx_node_name = "/Conv"} : (f32, tensor<1x1x3x3xf32>, none) -> tensor<1x1x3x3xf32>
}) : () -> ()
