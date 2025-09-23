// RUN: xdsl-opt -p test-transform-dialect-erase-schedule %s | filecheck

func.func @hello() {
    return
}

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

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @hello() {
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    builtin.module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    }
// CHECK-NEXT:    builtin.module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    }
// CHECK-NEXT:  }
