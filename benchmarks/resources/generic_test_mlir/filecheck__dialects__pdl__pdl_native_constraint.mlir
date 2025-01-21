"builtin.module"() ({
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "nativeConstraint"}> ({
    %0 = "pdl.type"() : () -> !pdl.type
    %1 = "pdl.operand"() : () -> !pdl.value
    %2 = "pdl.attribute"() : () -> !pdl.attribute
    %3 = "pdl.operation"(%1, %2, %0) <{attributeValueNames = ["value1"], opName = "test.test", operandSegmentSizes = array<i32: 1, 1, 1>}> : (!pdl.value, !pdl.attribute, !pdl.type) -> !pdl.operation
    "pdl.apply_native_constraint"(%0, %1, %2) <{isNegated = false, name = "myConstraint"}> : (!pdl.type, !pdl.value, !pdl.attribute) -> ()
    "pdl.rewrite"(%3, %3) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 1>}> ({
    }) : (!pdl.operation, !pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
