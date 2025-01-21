"builtin.module"() ({
  "pdl.pattern"() <{benefit = 1 : i16}> ({
    %0 = "pdl.attribute"() <{value = 0 : i64}> : () -> !pdl.attribute
    %1 = "pdl.operation"(%0) <{attributeValueNames = ["attr"], opName = "test.op", operandSegmentSizes = array<i32: 0, 1, 0>}> : (!pdl.attribute) -> !pdl.operation
    "pdl.rewrite"(%1) <{operandSegmentSizes = array<i32: 1, 0>}> ({
      %2 = "pdl.attribute"() <{value = 1 : i64}> : () -> !pdl.attribute
      %3 = "pdl.operation"(%2) <{attributeValueNames = ["attr"], opName = "test.op", operandSegmentSizes = array<i32: 0, 1, 0>}> : (!pdl.attribute) -> !pdl.operation
      "pdl.replace"(%1, %3) <{operandSegmentSizes = array<i32: 1, 1, 0>}> : (!pdl.operation, !pdl.operation) -> ()
    }) : (!pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
