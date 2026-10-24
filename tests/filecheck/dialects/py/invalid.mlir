// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

%0 = py.const 0
// CHECK: Expected at most 1 keyword names, got 2
%1 = "py.call"(%0) <{callee = "f", keywords = ["a", "b"]}> : (!py.object) -> !py.object
