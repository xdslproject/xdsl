"builtin.module"() ({
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "replaceWithValues"}> ({
    %4 = "pdl.type"() : () -> !pdl.type
    %5 = "pdl.operand"(%4) : (!pdl.type) -> !pdl.value
    %6 = "pdl.operand"(%4) : (!pdl.type) -> !pdl.value
    %7 = "pdl.operation"(%5, %6, %4, %4) <{attributeValueNames = [], operandSegmentSizes = array<i32: 2, 0, 2>}> : (!pdl.value, !pdl.value, !pdl.type, !pdl.type) -> !pdl.operation
    "pdl.rewrite"(%7) <{operandSegmentSizes = array<i32: 1, 0>}> ({
      "pdl.replace"(%7, %5, %6) <{operandSegmentSizes = array<i32: 1, 0, 2>}> : (!pdl.operation, !pdl.value, !pdl.value) -> ()
    }) : (!pdl.operation) -> ()
  }) : () -> ()
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "replaceWithOp"}> ({
    %0 = "pdl.type"() : () -> !pdl.type
    %1 = "pdl.operation"(%0) <{attributeValueNames = [], operandSegmentSizes = array<i32: 0, 0, 1>}> : (!pdl.type) -> !pdl.operation
    %2 = "pdl.result"(%1) <{index = 0 : i32}> : (!pdl.operation) -> !pdl.value
    %3 = "pdl.operation"(%2, %0) <{attributeValueNames = [], operandSegmentSizes = array<i32: 1, 0, 1>}> : (!pdl.value, !pdl.type) -> !pdl.operation
    "pdl.rewrite"(%3) <{operandSegmentSizes = array<i32: 1, 0>}> ({
      "pdl.replace"(%3, %1) <{operandSegmentSizes = array<i32: 1, 1, 0>}> : (!pdl.operation, !pdl.operation) -> ()
    }) : (!pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
