"builtin.module"() ({
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "extractResult"}> ({
    %6 = "pdl.types"() : () -> !pdl.range<type>
    %7 = "pdl.operation"(%6) <{attributeValueNames = [], operandSegmentSizes = array<i32: 0, 0, 1>}> : (!pdl.range<type>) -> !pdl.operation
    %8 = "pdl.result"(%7) <{index = 1 : i32}> : (!pdl.operation) -> !pdl.value
    "pdl.rewrite"(%7) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 0>}> ({
    }) : (!pdl.operation) -> ()
  }) : () -> ()
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "extractAllResults"}> ({
    %3 = "pdl.types"() : () -> !pdl.range<type>
    %4 = "pdl.operation"(%3) <{attributeValueNames = [], operandSegmentSizes = array<i32: 0, 0, 1>}> : (!pdl.range<type>) -> !pdl.operation
    %5 = "pdl.results"(%4) : (!pdl.operation) -> !pdl.range<value>
    "pdl.rewrite"(%4) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 0>}> ({
    }) : (!pdl.operation) -> ()
  }) : () -> ()
  "pdl.pattern"() <{benefit = 1 : i16, sym_name = "extractOneResultRange"}> ({
    %0 = "pdl.types"() : () -> !pdl.range<type>
    %1 = "pdl.operation"(%0) <{attributeValueNames = [], operandSegmentSizes = array<i32: 0, 0, 1>}> : (!pdl.range<type>) -> !pdl.operation
    %2 = "pdl.results"(%1) <{index = 1 : i32}> : (!pdl.operation) -> !pdl.range<value>
    "pdl.rewrite"(%1) <{name = "test_rewriter", operandSegmentSizes = array<i32: 1, 0>}> ({
    }) : (!pdl.operation) -> ()
  }) : () -> ()
}) : () -> ()
