"builtin.module"() ({
  "test.op"() : () -> ()
  "pdl.pattern"() <{benefit = 42 : i16}> ({
    %0 = "pdl.operation"() <{attributeValueNames = [], opName = "test.op", operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> !pdl.operation
    "pdl.rewrite"(%0) <{operandSegmentSizes = array<i32: 1, 0>}> ({
      "pdl.erase"(%0) : (!pdl.operation) -> ()
    }) : (!pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
