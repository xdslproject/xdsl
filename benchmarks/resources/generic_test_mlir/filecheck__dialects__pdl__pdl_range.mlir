"builtin.module"() ({
  %0:2 = "test.op"() : () -> (!pdl.range<operation>, !pdl.range<attribute>)
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "emptyRanges"}> ({
    %11 = "pdl.operation"() <{attributeValueNames = [], operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> !pdl.operation
    "pdl.rewrite"(%11) <{operandSegmentSizes = array<i32: 1, 0>}> ({
      %12 = "pdl.range"() : () -> !pdl.range<type>
      %13 = "pdl.range"() : () -> !pdl.range<value>
      %14 = "pdl.operation"(%13, %12) <{attributeValueNames = [], opName = "test.op", operandSegmentSizes = array<i32: 1, 0, 1>}> : (!pdl.range<value>, !pdl.range<type>) -> !pdl.operation
    }) : (!pdl.operation) -> ()
  }) : () -> ()
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "nonEmptyRanges"}> ({
    %1 = "pdl.type"() <{constantType = !pdl.type}> : () -> !pdl.type
    %2 = "pdl.type"() <{constantType = !pdl.type}> : () -> !pdl.type
    %3 = "pdl.operand"(%1) : (!pdl.type) -> !pdl.value
    %4 = "pdl.operand"(%2) : (!pdl.type) -> !pdl.value
    %5 = "pdl.operation"(%3, %4) <{attributeValueNames = [], operandSegmentSizes = array<i32: 2, 0, 0>}> : (!pdl.value, !pdl.value) -> !pdl.operation
    "pdl.rewrite"(%5) <{operandSegmentSizes = array<i32: 1, 0>}> ({
      %6 = "pdl.range"(%1) : (!pdl.type) -> !pdl.range<type>
      %7 = "pdl.range"(%6, %2) : (!pdl.range<type>, !pdl.type) -> !pdl.range<type>
      %8 = "pdl.range"(%3) : (!pdl.value) -> !pdl.range<value>
      %9 = "pdl.range"(%8, %4) : (!pdl.range<value>, !pdl.value) -> !pdl.range<value>
      %10 = "pdl.operation"(%9, %7) <{attributeValueNames = [], opName = "test.op", operandSegmentSizes = array<i32: 1, 0, 1>}> : (!pdl.range<value>, !pdl.range<type>) -> !pdl.operation
    }) : (!pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
