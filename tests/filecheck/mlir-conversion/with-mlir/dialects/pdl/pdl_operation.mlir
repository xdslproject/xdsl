// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

pdl.pattern @unboundedOperation : benefit(1) {
  // An unbounded operation
  %root = pdl.operation

  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @unboundedOperation
// CHECK: %{{\d+}} = pdl.operation

pdl.pattern @boundedOperation : benefit(1) {
  %type = pdl.type
  %types = pdl.types

  %operand = pdl.operand
  %operands = pdl.operands

  %attr = pdl.attribute
  %attr2 = pdl.attribute

  // A bound operation
  %root = pdl.operation "test.test"(%operand, %operands : !pdl.value, !pdl.range<value>)
                        {"value1" = %attr, "value2" = %attr2}
                        -> (%types, %type : !pdl.range<type>, !pdl.type)

  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @boundedOperation
// CHECK: %{{\d+}} = pdl.operation "test.test" (%{{\d+}}, %{{\d+}} : !pdl.value, !pdl.range<value>) {"value1" = %{{\d+}}, "value2" = %{{\d+}}} -> (%{{\d+}}, %{{\d+}} : !pdl.range<type>, !pdl.type)

pdl.pattern @noresultsOperation : benefit(1) {
    %root = pdl.operation "test.op"
    pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK:       pdl.pattern @noresultsOperation : benefit(1) {
// CHECK-NEXT:     %{{\d+}} = pdl.operation "test.op"
// CHECK-NEXT:     pdl.rewrite %{{\d+}} with "test_rewriter"(%{{\d+}} : !pdl.operation)
// CHECK-NEXT:  }
