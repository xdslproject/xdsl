// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

builtin.module {
  func.func @mixed_args(%0 : !test.type<"int">, i32) -> !test.type<"int"> {
  ^0(%arg : !test.type<"int">)
    %1 = "func.call"(%0) {"callee" = @cus_arg_rec} : (!test.type<"int">) -> !test.type<"int">
    "func.return"(%1) : (!test.type<"int">) -> ()
  }
}

// CHECK: Expected all arguments to be named or all arguments to be unnamed.
