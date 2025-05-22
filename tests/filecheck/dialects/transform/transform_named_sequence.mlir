// RUN: xdsl-opt %s -p transform-interpreter | filecheck %s


module {

  func.func @foo() -> i32 {
    %c1 = arith.constant 1 : i32
    %add = arith.addi %c1, %c1 : i32
    return %add : i32
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0 : !transform.op<"builtin.module">) {
      %0 = transform.apply_registered_pass "canonicalize" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
      transform.yield
    }
  }
}

// CHECK: func.func @foo() -> i32 {
// CHECK-NEXT: %add = arith.constant 2 : i32
// CHECK-NEXT: func.return %add : i32
// CHECK-NEXT: }

// CHECK:      transform.named_sequence @__transform_main(%arg0 : !transform.op<"builtin.module">) {
// CHECK-NEXT:   %0 = transform.apply_registered_pass "canonicalize" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
// CHECK-NEXT:   transform.yield
// CHECK-NEXT: }
