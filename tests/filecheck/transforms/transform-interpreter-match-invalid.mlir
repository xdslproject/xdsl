// RUN: xdsl-opt %s -p transform-interpreter --split-input-file --verify-diagnostics | filecheck %s

// Reject the unsupported interface{LinalgOp} filter.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match interface{LinalgOp} in %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match does not yet support the interface filter{{$}}

// -----

// Reject the unsupported interface{TilingInterface} filter.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match interface{TilingInterface} in %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match does not yet support the interface filter{{$}}

// -----

// Reject the unsupported interface{LoopLikeInterface} filter.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match interface{LoopLikeInterface} in %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match does not yet support the interface filter{{$}}

// -----

// Reject the unsupported attributes {} filter.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match attributes {} in %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match does not yet support the op_attrs filter{{$}}

// -----

// Reject the unsupported attributes {foo} filter.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match attributes {foo} in %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match does not yet support the op_attrs filter{{$}}

// -----

// Reject the unsupported filter_result_type = i32 filter.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match filter_result_type = i32 in %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match does not yet support the filter_result_type filter{{$}}

// -----

// Reject the unsupported filter_operand_types = [] filter.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match filter_operand_types = [] in %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match does not yet support the filter_operand_types filter{{$}}

// -----

// Reject the unsupported filter_operand_types = [i32] filter.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match filter_operand_types = [i32] in %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match does not yet support the filter_operand_types filter{{$}}

// -----

// Reject the unsupported interface{LinalgOp} filter, even when the name filter matches nothing.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{[]} interface{LinalgOp} in %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match does not yet support the interface filter{{$}}

// -----

// Reject the unsupported interface{TilingInterface} filter, even when the name filter matches nothing.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{[]} interface{TilingInterface} in %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match does not yet support the interface filter{{$}}

// -----

// Reject the unsupported interface{LoopLikeInterface} filter, even when the name filter matches nothing.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{[]} interface{LoopLikeInterface} in %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match does not yet support the interface filter{{$}}

// -----

// Reject the unsupported attributes {} filter, even when the name filter matches nothing.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{[]} attributes {} in %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match does not yet support the op_attrs filter{{$}}

// -----

// Reject the unsupported attributes {foo} filter, even when the name filter matches nothing.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{[]} attributes {foo} in %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match does not yet support the op_attrs filter{{$}}

// -----

// Reject the unsupported filter_result_type = i32 filter, even when the name filter matches nothing.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{[]} filter_result_type = i32 in %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match does not yet support the filter_result_type filter{{$}}

// -----

// Reject the unsupported filter_operand_types = [] filter, even when the name filter matches nothing.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{[]} filter_operand_types = [] in %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match does not yet support the filter_operand_types filter{{$}}

// -----

// Reject the unsupported filter_operand_types = [i32] filter, even when the name filter matches nothing.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matched = transform.structured.match ops{[]} filter_operand_types = [i32] in %root : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match does not yet support the filter_operand_types filter{{$}}

// -----

// A successful empty match cannot be used as the root of another match.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %empty = transform.structured.match ops{[]} in %root : (!transform.any_op) -> !transform.any_op
    %matched = transform.structured.match in %empty : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match requires exactly one target operation{{$}}

// -----

// A handle containing two operations cannot be used as a match root either.
module attributes {transform.with_named_sequence} {
  func.func private @first()
  func.func private @second()
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %functions = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %matched = transform.structured.match in %functions : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.match requires exactly one target operation{{$}}
