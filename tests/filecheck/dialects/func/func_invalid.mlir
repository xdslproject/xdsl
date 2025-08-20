// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

builtin.module {
  func.func @cus_arg_rec(%0 : !test.type<"int">) -> !test.type<"int"> {
  ^bb0(%arg : !test.type<"int">)
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
  ^bb0(%arg : !test.type<"int">)
    %1 = "func.call"(%0) {"callee" = @cus_arg_rec} : (!test.type<"int">) -> !test.type<"int">
    "func.return"(%1) : (!test.type<"int">) -> ()
  }
}

// CHECK: Expected all arguments to be named or all arguments to be unnamed.

// -----

"func.call"() { "callee" = @call::@invalid } : () -> ()

// CHECK:       Operation does not verify: Expected SymbolRefAttr with no nested symbols.
// CHECK-NEXT:  Underlying verification failure: expected empty array, but got ["invalid"]

// -----

func.func @bar() {
    %1 = "test.op"() : () -> !test.type<"int">
    %2 = func.call @foo(%1) : (!test.type<"int">) -> !test.type<"int">
    func.return
}

// CHECK: '@foo' could not be found in symbol table

// -----

func.func @foo(%0 : !test.type<"int">) -> !test.type<"int">

func.func @bar() {
    %1 = func.call @foo() : () -> !test.type<"int">
    func.return
}

// CHECK: incorrect number of operands for callee

// -----

func.func @foo(%0 : !test.type<"int">)

func.func @bar() {
    %1 = "test.op"() : () -> !test.type<"int">
    %2 = func.call @foo(%1) : (!test.type<"int">) -> !test.type<"int">
    func.return
}

// CHECK: incorrect number of results for callee

// -----

func.func @foo(%0 : !test.type<"int">) -> !test.type<"int">

func.func @bar() {
  %1 = "test.op"() : () -> !test.type<"foo">
  %2 = func.call @foo(%1) : (!test.type<"foo">) -> !test.type<"int">
  func.return
}

// CHECK: operand type mismatch: expected operand type !test.type<"int">, but provided !test.type<"foo"> for operand number 0

// -----

func.func @foo(%0 : !test.type<"int">) -> !test.type<"int">

func.func @bar() {
    %1 = "test.op"() : () -> !test.type<"int">
    %2 = func.call @foo(%1) : (!test.type<"int">) -> !test.type<"foo">
    func.return
}

// CHECK: result type mismatch: expected result type !test.type<"int">, but provided !test.type<"foo"> for result number 0
