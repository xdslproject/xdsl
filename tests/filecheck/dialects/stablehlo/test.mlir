// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

builtin.module {
  func.func @different_shapes(%0: tensor<i32>) {
    // CHECK: "stablehlo.abs"(%{{.*}}) : (tensor<i32>) -> tensor<i32>
    %result = "stablehlo.abs"(%0) : (tensor<i32>) -> tensor<i32>
    func.return
  }
}
