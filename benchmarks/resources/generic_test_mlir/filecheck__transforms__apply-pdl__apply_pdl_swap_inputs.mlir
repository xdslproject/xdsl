"builtin.module"() ({
  "func.func"() <{function_type = () -> i32, sym_name = "impl"}> ({
    %9 = "arith.constant"() <{value = 4 : i32}> : () -> i32
    %10 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %12 = "arith.addi"(%9, %10) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %13 = "arith.addi"(%12, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "func.return"(%13) : (i32) -> ()
  }) : () -> ()
  "pdl.pattern"() <{benefit = 2 : i16}> ({
    %0 = "pdl.operand"() : () -> !pdl.value
    %1 = "pdl.operand"() : () -> !pdl.value
    %2 = "pdl.type"() : () -> !pdl.type
    %3 = "pdl.operation"(%0, %1, %2) <{attributeValueNames = [], opName = "arith.addi", operandSegmentSizes = array<i32: 2, 0, 1>}> : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation
    %4 = "pdl.result"(%3) <{index = 0 : i32}> : (!pdl.operation) -> !pdl.value
    %5 = "pdl.operand"() : () -> !pdl.value
    %6 = "pdl.attribute"() : () -> !pdl.attribute
    %7 = "pdl.operation"(%4, %5, %6, %2) <{attributeValueNames = ["overflowFlags"], opName = "arith.addi", operandSegmentSizes = array<i32: 2, 1, 1>}> : (!pdl.value, !pdl.value, !pdl.attribute, !pdl.type) -> !pdl.operation
    "pdl.rewrite"(%7) <{operandSegmentSizes = array<i32: 1, 0>}> ({
      %8 = "pdl.operation"(%5, %4, %6, %2) <{attributeValueNames = ["overflowFlags"], opName = "arith.addi", operandSegmentSizes = array<i32: 2, 1, 1>}> : (!pdl.value, !pdl.value, !pdl.attribute, !pdl.type) -> !pdl.operation
      "pdl.replace"(%7, %8) <{operandSegmentSizes = array<i32: 1, 1, 0>}> : (!pdl.operation, !pdl.operation) -> ()
    }) : (!pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
