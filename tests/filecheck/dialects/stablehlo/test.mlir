// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

builtin.module {
  %t0 = "test.op"() : () -> tensor<i32>

  // CHECK: "stablehlo.abs"(%{{.*}}) : (tensor<i32>) -> tensor<i32>
  %r0 = "stablehlo.abs"(%t0) : (tensor<i32>) -> tensor<i32>
}
