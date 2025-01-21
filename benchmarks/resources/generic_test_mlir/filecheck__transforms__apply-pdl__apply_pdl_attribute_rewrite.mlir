"builtin.module"() ({
  %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  "pdl.pattern"() <{benefit = 2 : i16}> ({
    %1 = "pdl.type"() : () -> !pdl.type
    %2 = "pdl.attribute"() <{value = 0 : i32}> : () -> !pdl.attribute
    %3 = "pdl.operation"(%2, %1) <{attributeValueNames = ["value"], opName = "arith.constant", operandSegmentSizes = array<i32: 0, 1, 1>}> : (!pdl.attribute, !pdl.type) -> !pdl.operation
    "pdl.rewrite"(%3) <{operandSegmentSizes = array<i32: 1, 0>}> ({
      %4 = "pdl.attribute"() <{value = 1 : i32}> : () -> !pdl.attribute
      %5 = "pdl.operation"(%4, %1) <{attributeValueNames = ["value"], opName = "arith.constant", operandSegmentSizes = array<i32: 0, 1, 1>}> : (!pdl.attribute, !pdl.type) -> !pdl.operation
      "pdl.replace"(%3, %5) <{operandSegmentSizes = array<i32: 1, 1, 0>}> : (!pdl.operation, !pdl.operation) -> ()
    }) : (!pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
