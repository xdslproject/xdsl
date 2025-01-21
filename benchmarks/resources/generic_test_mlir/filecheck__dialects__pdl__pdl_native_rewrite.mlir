"builtin.module"() ({
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "nativeRewrite"}> ({
    %0 = "pdl.type"() : () -> !pdl.type
    %1 = "pdl.operand"() : () -> !pdl.value
    %2 = "pdl.attribute"() : () -> !pdl.attribute
    %3 = "pdl.operation"(%1, %2, %0) <{attributeValueNames = ["value1"], opName = "test.test", operandSegmentSizes = array<i32: 1, 1, 1>}> : (!pdl.value, !pdl.attribute, !pdl.type) -> !pdl.operation
    "pdl.rewrite"(%3) <{operandSegmentSizes = array<i32: 1, 0>}> ({
      %4 = "pdl.apply_native_rewrite"(%0, %1, %2) <{name = "myRewrite"}> : (!pdl.type, !pdl.value, !pdl.attribute) -> !pdl.operation
      "pdl.replace"(%3, %4) <{operandSegmentSizes = array<i32: 1, 1, 0>}> : (!pdl.operation, !pdl.operation) -> ()
    }) : (!pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
