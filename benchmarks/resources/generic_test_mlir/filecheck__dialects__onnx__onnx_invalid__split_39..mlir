"builtin.module"() ({
  %0 = "test.op"() : () -> tensor<5x5x32x32xf32>
  %1 = "onnx.MaxPoolSingleOut"(%0) {onnx_node_name = "/MaxPoolSingleOut"} : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xi32>
}) : () -> ()
