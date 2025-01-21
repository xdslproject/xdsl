"builtin.module"() ({
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "unboundedOperand"}> ({
    %8 = "pdl.operand"() : () -> !pdl.value
    %9 = "pdl.operation"(%8) <{attributeValueNames = [], operandSegmentSizes = array<i32: 1, 0, 0>}> : (!pdl.value) -> !pdl.operation
    "pdl.rewrite"(%9, %9) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 1>}> ({
    }) : (!pdl.operation, !pdl.operation) -> ()
  }) : () -> ()
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "boundedOperand"}> ({
    %5 = "pdl.type"() <{constantType = i32}> : () -> !pdl.type
    %6 = "pdl.operand"(%5) : (!pdl.type) -> !pdl.value
    %7 = "pdl.operation"(%6) <{attributeValueNames = [], operandSegmentSizes = array<i32: 1, 0, 0>}> : (!pdl.value) -> !pdl.operation
    "pdl.rewrite"(%7, %7) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 1>}> ({
    }) : (!pdl.operation, !pdl.operation) -> ()
  }) : () -> ()
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "unboundedOperands"}> ({
    %3 = "pdl.operands"() : () -> !pdl.range<value>
    %4 = "pdl.operation"(%3) <{attributeValueNames = [], operandSegmentSizes = array<i32: 1, 0, 0>}> : (!pdl.range<value>) -> !pdl.operation
    "pdl.rewrite"(%4, %4) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 1>}> ({
    }) : (!pdl.operation, !pdl.operation) -> ()
  }) : () -> ()
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "boundedOperands"}> ({
    %0 = "pdl.types"() <{constantTypes = [i32, i64, i1]}> : () -> !pdl.range<type>
    %1 = "pdl.operands"(%0) : (!pdl.range<type>) -> !pdl.range<value>
    %2 = "pdl.operation"(%1) <{attributeValueNames = [], operandSegmentSizes = array<i32: 1, 0, 0>}> : (!pdl.range<value>) -> !pdl.operation
    "pdl.rewrite"(%2, %2) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 1>}> ({
    }) : (!pdl.operation, !pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
