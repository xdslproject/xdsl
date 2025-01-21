"builtin.module"() ({
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "unboundedAttribute"}> ({
    %5 = "pdl.attribute"() : () -> !pdl.attribute
    %6 = "pdl.operation"(%5) <{attributeValueNames = ["attr"], operandSegmentSizes = array<i32: 0, 1, 0>}> : (!pdl.attribute) -> !pdl.operation
    "pdl.rewrite"(%6, %6) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 1>}> ({
    }) : (!pdl.operation, !pdl.operation) -> ()
  }) : () -> ()
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "typedAttribute"}> ({
    %2 = "pdl.type"() <{constantType = i32}> : () -> !pdl.type
    %3 = "pdl.attribute"(%2) : (!pdl.type) -> !pdl.attribute
    %4 = "pdl.operation"(%3) <{attributeValueNames = ["attr"], operandSegmentSizes = array<i32: 0, 1, 0>}> : (!pdl.attribute) -> !pdl.operation
    "pdl.rewrite"(%4, %4) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 1>}> ({
    }) : (!pdl.operation, !pdl.operation) -> ()
  }) : () -> ()
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "constantAttribute"}> ({
    %0 = "pdl.attribute"() <{value = 0 : i32}> : () -> !pdl.attribute
    %1 = "pdl.operation"(%0) <{attributeValueNames = ["attr"], operandSegmentSizes = array<i32: 0, 1, 0>}> : (!pdl.attribute) -> !pdl.operation
    "pdl.rewrite"(%1, %1) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 1>}> ({
    }) : (!pdl.operation, !pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
