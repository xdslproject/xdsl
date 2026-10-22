// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

%0 = py.const 0
// CHECK: Expected at most 1 keyword names, got 2
%1 = "py.call"(%0) <{callee = "f", keywords = ["a", "b"]}> : (!py.object) -> !py.object

// -----

// CHECK: Expected 2 block arguments, one per annotation, got 1
"py.func"() <{sym_name = "f", arg_annotations = ["int", "int"], return_annotation = "None"}> ({
^bb0(%x : !py.object):
  py.return
}) : () -> ()

// -----

// CHECK: Expected all block arguments to be !py.object
"py.func"() <{sym_name = "f", arg_annotations = ["int"], return_annotation = "None"}> ({
^bb0(%x : i32):
  py.return
}) : () -> ()

// -----

%0 = py.const 0
// CHECK: 'py.return' expects parent op 'py.func'
py.return %0
