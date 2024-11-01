// RUN: XDSL_ROUNDTRIP

pdl.pattern @unboundedOperation : benefit(1) {
  // An unbounded operation
  %root = pdl.operation

  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @unboundedOperation
// CHECK: %{{.*}} = pdl.operation

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
// CHECK: %{{.*}} = pdl.operation "test.test" (%{{.*}}, %{{.*}} : !pdl.value, !pdl.range<value>) {"value1" = %{{.*}}, "value2" = %{{.*}}} -> (%{{.*}}, %{{.*}} : !pdl.range<type>, !pdl.type)

pdl.pattern @noresultsOperation : benefit(1) {
    %root = pdl.operation "test.op"
    pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK:       pdl.pattern @noresultsOperation : benefit(1) {
// CHECK-NEXT:     %root_2 = pdl.operation "test.op"
// CHECK-NEXT:     pdl.rewrite %root_2 with "test_rewriter"(%root_2 : !pdl.operation)
// CHECK-NEXT:  }
// CHECK-NEXT:   
