// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

pdl.pattern @unboundedOperand : benefit(1) {
  // An unbounded operand
  %operand = pdl.operand

  %root = pdl.operation(%operand : !pdl.value)
  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @unboundedOperand
// CHECK: %{{\d+}} = pdl.operand

pdl.pattern @boundedOperand : benefit(1) {
  // An operand with a known type
  %type = pdl.type : i32
  %operand = pdl.operand : %type

  %root = pdl.operation(%operand : !pdl.value)
  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @boundedOperand
// CHECK: %{{\d+}} = pdl.operand : %{{\d+}}

pdl.pattern @unboundedOperands : benefit(1) {
  // Unbounded operands
  %operands = pdl.operands

  %root = pdl.operation(%operands : !pdl.range<value>)
  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @unboundedOperands
// CHECK: %{{\d+}} = pdl.operands

pdl.pattern @boundedOperands : benefit(1) {
  // Operands with known types
  %types = pdl.types : [i32, i64, i1]
  %operands = pdl.operands : %types

  %root = pdl.operation(%operands : !pdl.range<value>)
  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @boundedOperands
// CHECK: %{{\d+}} = pdl.operands : %{{\d+}}
