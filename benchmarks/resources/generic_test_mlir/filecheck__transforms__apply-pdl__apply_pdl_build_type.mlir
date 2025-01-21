"builtin.module"() ({
  %0 = "test.op"() : () -> i32
  "pdl.pattern"() <{benefit = 1 : i16}> ({
    %1 = "pdl.type"() <{constantType = i32}> : () -> !pdl.type
    %2 = "pdl.operation"(%1) <{attributeValueNames = [], opName = "test.op", operandSegmentSizes = array<i32: 0, 0, 1>}> : (!pdl.type) -> !pdl.operation
    "pdl.rewrite"(%2) <{operandSegmentSizes = array<i32: 1, 0>}> ({
      %3 = "pdl.type"() <{constantType = i64}> : () -> !pdl.type
      %4 = "pdl.operation"(%3) <{attributeValueNames = [], opName = "test.op", operandSegmentSizes = array<i32: 0, 0, 1>}> : (!pdl.type) -> !pdl.operation
      "pdl.replace"(%2, %4) <{operandSegmentSizes = array<i32: 1, 1, 0>}> : (!pdl.operation, !pdl.operation) -> ()
    }) : (!pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
