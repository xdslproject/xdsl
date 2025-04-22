// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

builtin.module {
  func.func @foo() {
    func.return
  }

  builtin.module attributes {transform.with_named_sequence} {
    "transform.named_sequence"() <{sym_name = "__transform_main", function_type = (!transform.op<"builtin.module">) -> ()}> ({
      ^0(%arg0 : !transform.op<"builtin.module">):
        // CHECK: transform.apply_registered_pass "foo" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        %0 = "transform.apply_registered_pass"(%arg0) {pass_name = "foo"} : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
        "transform.yield"() : () -> ()
    }) : () -> ()
  }
}
