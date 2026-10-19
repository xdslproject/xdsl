// RUN: xdsl-opt %s -p transform-interpreter | filecheck %s

// Print the result of each match to test execution, not just the output IR.
// Capture each operation's representation once and reuse it to check identities
// and order without depending on memory addresses. The regex stops at the end
// of each operation (">" preceded by ")") so extra matches cannot be hidden.
module attributes {transform.with_named_sequence} {
  // This constant is outside the selected payload and must not be matched.
  %outside = arith.constant 3 : i32

  func.func @payload() {
    module {
      %inner = arith.constant 1 : i32
    }
    %outer = arith.constant 2 : i32
    func.return
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %payload = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op

    // An absent filter matches everything under the payload, including the
    // payload itself. Children precede their parents in post-order traversal.
    %all = transform.structured.match in %payload : (!transform.any_op) -> !transform.any_op
    printf.print_format "all: {}\n", %all : !transform.any_op

    // An explicitly empty name filter matches nothing.
    %empty = transform.structured.match ops{[]} in %payload : (!transform.any_op) -> !transform.any_op
    printf.print_format "empty: {}\n", %empty : !transform.any_op

    // A name absent from the payload also yields an empty handle.
    %missing = transform.structured.match ops{["linalg.matmul"]} in %payload : (!transform.any_op) -> !transform.any_op
    printf.print_format "missing: {}\n", %missing : !transform.any_op

    // The root operation participates in name matching.
    %function = transform.structured.match ops{["func.func"]} in %payload : (!transform.any_op) -> !transform.any_op
    printf.print_format "function: {}\n", %function : !transform.any_op

    // Matching traverses nested regions and can return multiple operations.
    %constants = transform.structured.match ops{["arith.constant"]} in %payload : (!transform.any_op) -> !transform.any_op
    printf.print_format "constants: {}\n", %constants : !transform.any_op

    %modules = transform.structured.match ops{["builtin.module"]} in %payload : (!transform.any_op) -> !transform.any_op
    printf.print_format "modules: {}\n", %modules : !transform.any_op

    // Multiple names form a union in traversal order, not name-list order.
    %union = transform.structured.match ops{["builtin.module", "arith.constant"]} in %payload : (!transform.any_op) -> !transform.any_op
    printf.print_format "union: {}\n", %union : !transform.any_op

    // Repeating a name must not duplicate the matching operations.
    %duplicates = transform.structured.match ops{["arith.constant", "arith.constant"]} in %payload : (!transform.any_op) -> !transform.any_op
    printf.print_format "duplicates: {}\n", %duplicates : !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}all: OperationHandle(ops=([[INNER:<ConstantOp [0-9]+\(([^>]|[^)]>)*\)>]], [[MODULE:<ModuleOp [0-9]+\(([^>]|[^)]>)*\)>]], [[OUTER:<ConstantOp [0-9]+\(([^>]|[^)]>)*\)>]], [[RETURN:<ReturnOp [0-9]+\(([^>]|[^)]>)*\)>]], [[FUNCTION:<FuncOp [0-9]+\(([^>]|[^)]>)*\)>]])){{$}}
// CHECK-NEXT: empty: OperationHandle(ops=())
// CHECK-NEXT: missing: OperationHandle(ops=())
// CHECK-NEXT: function: OperationHandle(ops=([[FUNCTION]],))
// CHECK-NEXT: constants: OperationHandle(ops=([[INNER]], [[OUTER]]))
// CHECK-NEXT: modules: OperationHandle(ops=([[MODULE]],))
// CHECK-NEXT: union: OperationHandle(ops=([[INNER]], [[MODULE]], [[OUTER]]))
// CHECK-NEXT: duplicates: OperationHandle(ops=([[INNER]], [[OUTER]]))
