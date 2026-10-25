// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

%0 = py.const 0
// CHECK: Expected at most 1 keyword names, got 2
%1 = "py.call"(%0) <{callee = "f", keywords = ["a", "b"]}> : (!py.object) -> !py.object

// -----

// CHECK: Expected a single result type, got 0
py.func @f : () -> () {
  %0 = py.const 0
  py.return %0
}

// -----

// CHECK: Expected a function type of !py.object types
py.func @f : (i32) -> !py.object {
^bb0(%x : i32):
  %0 = py.const 0
  py.return %0
}

// -----

// CHECK: Expected the block arguments to match the function type inputs
py.func @f : (!py.object<"int">) -> !py.object {
^bb0(%x : !py.object):
  py.return %x
}

// -----

%0 = py.const 0
// CHECK: 'py.return' expects parent op 'py.func'
py.return %0
