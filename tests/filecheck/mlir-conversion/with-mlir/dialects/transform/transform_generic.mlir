// RUN: xdsl-opt %s --print-op-generic | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | filecheck %s

%0 = "test.op"() : () -> !transform.op<"builtin.module">
// CHECK: %1 = "transform.apply_registered_pass"(%0) <{options = {}, pass_name = "foo"}> : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
%1 = "transform.apply_registered_pass"(%0) <{pass_name = "foo", options = {}}> : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
// CHECK: %2 = "transform.apply_registered_pass"(%0) <{options = {hello = "bye"}, pass_name = "foo"}> : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
%2 = "transform.apply_registered_pass"(%0) <{pass_name = "foo"}> {options = {hello = "bye"}} : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
