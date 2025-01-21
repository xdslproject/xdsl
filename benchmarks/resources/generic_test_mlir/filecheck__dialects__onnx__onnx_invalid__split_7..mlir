"builtin.module"() ({
  %0:3 = "test.op"() : () -> (tensor<5x2xf32>, tensor<2x1xf32>, tensor<5x4xf32>)
  %1 = "onnx.Gemm"(%0#0, %0#1, %0#2) : (tensor<5x2xf32>, tensor<2x1xf32>, tensor<5x4xf32>) -> tensor<5x2xf32>
}) : () -> ()
