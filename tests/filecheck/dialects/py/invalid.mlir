// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

%0 = py.const 0
// CHECK: Expected at most 1 keyword names, got 2
%1 = "py.call"(%0) <{callee = "f", keywords = ["a", "b"]}> : (!py.object) -> !py.object

// -----

// CHECK: Expected a single result type, got 0
py.func @f() -> () {
  %0 = py.const 0
  py.return %0 : !py.object
}

// -----

// CHECK: Expected a function type of !py.object types
py.func @f(%x : i32) -> !py.object {
  %0 = py.const 0
  py.return %0 : !py.object
}

// -----

// CHECK: Expected the block arguments to match the function type inputs
"py.func"() <{func_name = "f", function_type = (!py.object<"int">) -> !py.object}> ({
^bb0(%x : !py.object):
  py.return %x : !py.object
}) : () -> ()

// -----

%0 = py.const 0
// CHECK: 'py.return' expects parent op 'py.func'
py.return %0 : !py.object

// -----

// CHECK: Expected the returned value to have the function's result type !py.object<"int">, got !py.object
py.func @f() -> !py.object<"int"> {
  %0 = py.const 0
  py.return %0 : !py.object
}

// -----

// CHECK: py.func does not take argument or result attributes
py.func @f(%x : !py.object {test = "a"}) -> !py.object {
  py.return %x : !py.object
}
