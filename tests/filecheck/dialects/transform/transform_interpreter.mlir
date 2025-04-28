// RUN: xdsl-opt %s -p transform-interpreter | filecheck %s
// RUN: xdsl-opt %s -p transform-interpreter'{entry-point=entry}' | filecheck %s


module {

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
      transform.yield
    }
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @entry(%arg0 : !transform.any_op {transform.readonly}) {
      transform.yield
    }
  }
}

// CHECK: transform.named_sequence @__transform_main(%arg0 : !transform.any_op {transform.readonly}) {
// CHECK-NEXT:   transform.yield
// CHECK-NEXT: }

// CHECK: transform.named_sequence @entry(%arg0 : !transform.any_op {transform.readonly}) {
// CHECK-NEXT:   transform.yield
// CHECK-NEXT: }
