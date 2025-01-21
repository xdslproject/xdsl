"builtin.module"() ({
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "unboundedType"}> ({
    %6 = "pdl.type"() : () -> !pdl.type
    %7 = "pdl.operation"(%6) <{attributeValueNames = [], operandSegmentSizes = array<i32: 0, 0, 1>}> : (!pdl.type) -> !pdl.operation
    "pdl.rewrite"(%7, %7) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 1>}> ({
    }) : (!pdl.operation, !pdl.operation) -> ()
  }) : () -> ()
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "knownType"}> ({
    %4 = "pdl.type"() <{constantType = i32}> : () -> !pdl.type
    %5 = "pdl.operation"(%4) <{attributeValueNames = [], operandSegmentSizes = array<i32: 0, 0, 1>}> : (!pdl.type) -> !pdl.operation
    "pdl.rewrite"(%5, %5) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 1>}> ({
    }) : (!pdl.operation, !pdl.operation) -> ()
  }) : () -> ()
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "unboundedTypes"}> ({
    %2 = "pdl.types"() : () -> !pdl.range<type>
    %3 = "pdl.operation"(%2) <{attributeValueNames = [], operandSegmentSizes = array<i32: 0, 0, 1>}> : (!pdl.range<type>) -> !pdl.operation
    "pdl.rewrite"(%3, %3) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 1>}> ({
    }) : (!pdl.operation, !pdl.operation) -> ()
  }) : () -> ()
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "knownTypes"}> ({
    %0 = "pdl.types"() <{constantTypes = [i32, i64]}> : () -> !pdl.range<type>
    %1 = "pdl.operation"(%0) <{attributeValueNames = [], operandSegmentSizes = array<i32: 0, 0, 1>}> : (!pdl.range<type>) -> !pdl.operation
    "pdl.rewrite"(%1, %1) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 1>}> ({
    }) : (!pdl.operation, !pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
