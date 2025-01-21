"builtin.module"() ({
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "unboundedOperation"}> ({
    %8 = "pdl.operation"() <{attributeValueNames = [], operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> !pdl.operation
    "pdl.rewrite"(%8, %8) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 1>}> ({
    }) : (!pdl.operation, !pdl.operation) -> ()
  }) : () -> ()
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "boundedOperation"}> ({
    %1 = "pdl.type"() : () -> !pdl.type
    %2 = "pdl.types"() : () -> !pdl.range<type>
    %3 = "pdl.operand"() : () -> !pdl.value
    %4 = "pdl.operands"() : () -> !pdl.range<value>
    %5 = "pdl.attribute"() : () -> !pdl.attribute
    %6 = "pdl.attribute"() : () -> !pdl.attribute
    %7 = "pdl.operation"(%3, %4, %5, %6, %2, %1) <{attributeValueNames = ["value1", "value2"], opName = "test.test", operandSegmentSizes = array<i32: 2, 2, 2>}> : (!pdl.value, !pdl.range<value>, !pdl.attribute, !pdl.attribute, !pdl.range<type>, !pdl.type) -> !pdl.operation
    "pdl.rewrite"(%7, %7) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 1>}> ({
    }) : (!pdl.operation, !pdl.operation) -> ()
  }) : () -> ()
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "noresultsOperation"}> ({
    %0 = "pdl.operation"() <{attributeValueNames = [], opName = "test.op", operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> !pdl.operation
    "pdl.rewrite"(%0, %0) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 1>}> ({
    }) : (!pdl.operation, !pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
