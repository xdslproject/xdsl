"builtin.module"() ({
  %0 = "test.op"() : () -> f32
  %1 = "onnx.MaxPoolSingleOut"(%0) {onnx_node_name = "/MaxPoolSingleOut"} : (f32) -> tensor<5x5x32x32xf32>
}) : () -> ()
