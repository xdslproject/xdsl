// RUN: XDSL_ROUNDTRIP

pdl.pattern @unboundedOperand : benefit(1) {
  // An unbounded operand
  %operand = pdl.operand
  
  %root = pdl.operation(%operand : !pdl.value)
  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @unboundedOperand
// CHECK: %{{.*}} = pdl.operand

pdl.pattern @boundedOperand : benefit(1) {
  // An operand with a known type
  %type = pdl.type : i32
  %operand = pdl.operand : %type
  
  %root = pdl.operation(%operand : !pdl.value)
  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @boundedOperand
// CHECK: %{{.*}} = pdl.operand : %{{\S+}}

pdl.pattern @unboundedOperands : benefit(1) {
  // Unbounded operands
  %operands = pdl.operands
  
  %root = pdl.operation(%operands : !pdl.range<value>)
  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @unboundedOperands
// CHECK: %{{.*}} = pdl.operands

pdl.pattern @boundedOperands : benefit(1) {
  // Operands with known types
  %types = pdl.types : [i32, i64, i1]
  %operands = pdl.operands : %types
  
  %root = pdl.operation(%operands : !pdl.range<value>)
  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @boundedOperands
// CHECK: %{{.*}} = pdl.operands : %{{\S+}}
