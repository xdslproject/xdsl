// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

builtin.module {
  func.func @mixed_args("unexpected_token") {
    "func.return"() : () -> ()
  }
}

// CHECK: Expected argument or type
