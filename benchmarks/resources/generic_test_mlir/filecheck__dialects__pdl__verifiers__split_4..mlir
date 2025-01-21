"builtin.module"() ({
  "pdl.pattern"() <{benefit = 0 : i16, sym_name = "UnusedTypes"}> ({
    %0 = "pdl.types"() <{constantTypes = [i32, i32]}> : () -> !pdl.range<type>
    %1 = "pdl.operation"() <{attributeValueNames = [], opName = "test.op", operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> !pdl.operation
    "pdl.rewrite"(%1) <{operandSegmentSizes = array<i32: 1, 0>}> ({
      "pdl.erase"(%1) : (!pdl.operation) -> ()
    }) : (!pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
