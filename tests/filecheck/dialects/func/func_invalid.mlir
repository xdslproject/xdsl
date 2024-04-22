// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

builtin.module {
  func.func @cus_arg_rec(%0 : !test.type<"int">) -> !test.type<"int"> {
  ^0(%arg : !test.type<"int">)
    %1 = "func.call"(%0) {"callee" = @cus_arg_rec} : (!test.type<"int">) -> !test.type<"int">
    "func.return"(%1) : (!test.type<"int">) -> ()
  }
}

// CHECK: invalid block name in region with named arguments

// -----

builtin.module {
  func.func @mixed_args("unexpected_token") {
    "func.return"() : () -> ()
  }
}

// CHECK: Expected argument or type

// -----

builtin.module {
  func.func @mixed_args(%0 : !test.type<"int">, i32) -> !test.type<"int"> {
  ^0(%arg : !test.type<"int">)
    %1 = "func.call"(%0) {"callee" = @cus_arg_rec} : (!test.type<"int">) -> !test.type<"int">
    "func.return"(%1) : (!test.type<"int">) -> ()
  }
}

// CHECK: Expected all arguments to be named or all arguments to be unnamed.

// -----

"func.call"() { "callee" = @call::@invalid } : () -> ()

// CHECK:       Operation does not verify: Unexpected nested symbols in FlatSymbolRefAttr
// CHECK-NEXT:  Underlying verification failure: expected empty array, but got ["invalid"]
