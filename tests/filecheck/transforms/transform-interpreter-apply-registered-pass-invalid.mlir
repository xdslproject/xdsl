// RUN: xdsl-opt %s -p transform-interpreter --split-input-file --verify-diagnostics | filecheck %s

// Matching makes non-module handles available through the CLI. Registered
// passes must diagnose these targets because xDSL's pass API requires modules.
module attributes {transform.with_named_sequence} {
  func.func private @foo()
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %function = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %result = transform.apply_registered_pass "canonicalize" to %function : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.apply_registered_pass currently supports only builtin.module targets{{$}}

// -----

// Reject a mixed handle before modifying its valid targets. Post-order matching
// visits @payload before @foo; the unused constant must survive in the diagnostic.
module attributes {transform.with_named_sequence} {
  module @payload {
    %unused = arith.constant 42 : i32
  }
  func.func private @foo()
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mixed = transform.structured.match ops{["builtin.module", "func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %result = transform.apply_registered_pass "canonicalize" to %mixed : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.apply_registered_pass currently supports only builtin.module targets{{$}}
// CHECK: "arith.constant"() <{value = 42 : i32}>
