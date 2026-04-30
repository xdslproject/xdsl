builtin.module {
  func.func @softmax(%input: tensor<2x3xf32>, %output: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %r = linalg.softmax dimension(1) ins(%input : tensor<2x3xf32>) outs(%output : tensor<2x3xf32>) {acc_bound = 1.000000e-03 : f32} -> tensor<2x3xf32>
    func.return %r : tensor<2x3xf32>
  }
}
