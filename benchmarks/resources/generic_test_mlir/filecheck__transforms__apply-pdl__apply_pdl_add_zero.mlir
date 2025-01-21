"builtin.module"() ({
  "func.func"() <{function_type = () -> i32, sym_name = "impl"}> ({
    %6 = "arith.constant"() <{value = 4 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %8 = "arith.addi"(%6, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "func.return"(%8) : (i32) -> ()
  }) : () -> ()
  "pdl.pattern"() <{benefit = 2 : i16}> ({
    %0 = "pdl.type"() : () -> !pdl.type
    %1 = "pdl.operand"() : () -> !pdl.value
    %2 = "pdl.attribute"() <{value = 0 : i32}> : () -> !pdl.attribute
    %3 = "pdl.operation"(%2, %0) <{attributeValueNames = ["value"], opName = "arith.constant", operandSegmentSizes = array<i32: 0, 1, 1>}> : (!pdl.attribute, !pdl.type) -> !pdl.operation
    %4 = "pdl.result"(%3) <{index = 0 : i32}> : (!pdl.operation) -> !pdl.value
    %5 = "pdl.operation"(%1, %4, %0) <{attributeValueNames = [], opName = "arith.addi", operandSegmentSizes = array<i32: 2, 0, 1>}> : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation
    "pdl.rewrite"(%5) <{operandSegmentSizes = array<i32: 1, 0>}> ({
      "pdl.replace"(%5, %1) <{operandSegmentSizes = array<i32: 1, 0, 1>}> : (!pdl.operation, !pdl.value) -> ()
    }) : (!pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
