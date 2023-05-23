// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

builtin.module {
  func.func @cus_arg_rec(%0 : !test.type<"int">) -> !test.type<"int"> {
  ^0(%arg : !test.type<"int">)
    %1 = "func.call"(%0) {"callee" = @cus_arg_rec} : (!test.type<"int">) -> !test.type<"int">
    "func.return"(%1) : (!test.type<"int">) -> ()
  }
}

// CHECK: invalid block name in region with named arguments
