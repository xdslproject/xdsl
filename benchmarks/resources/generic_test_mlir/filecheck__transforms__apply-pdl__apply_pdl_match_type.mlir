"builtin.module"() ({
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %1 = "arith.constant"() <{value = 84 : i64}> : () -> i64
  "pdl.pattern"() <{benefit = 1 : i16}> ({
    %2 = "pdl.type"() <{constantType = i32}> : () -> !pdl.type
    %3 = "pdl.attribute"(%2) : (!pdl.type) -> !pdl.attribute
    %4 = "pdl.operation"(%3, %2) <{attributeValueNames = ["value"], opName = "arith.constant", operandSegmentSizes = array<i32: 0, 1, 1>}> : (!pdl.attribute, !pdl.type) -> !pdl.operation
    "pdl.rewrite"(%4) <{operandSegmentSizes = array<i32: 1, 0>}> ({
      "pdl.erase"(%4) : (!pdl.operation) -> ()
    }) : (!pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
