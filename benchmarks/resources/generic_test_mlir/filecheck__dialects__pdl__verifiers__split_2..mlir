"builtin.module"() ({
  "pdl.pattern"() <{benefit = 0 : i16, sym_name = "UnusedAttribute"}> ({
    %0 = "pdl.attribute"() <{value = 0 : i32}> : () -> !pdl.attribute
    %1 = "pdl.operation"() <{attributeValueNames = [], opName = "test.op", operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> !pdl.operation
    "pdl.rewrite"(%1) <{operandSegmentSizes = array<i32: 1, 0>}> ({
      "pdl.erase"(%1) : (!pdl.operation) -> ()
    }) : (!pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
