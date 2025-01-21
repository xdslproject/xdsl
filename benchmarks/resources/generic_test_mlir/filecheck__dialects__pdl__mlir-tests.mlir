"builtin.module"() ({
  "builtin.module"() ({
    "pdl.pattern"() <{benefit = 1 : i16, sym_name = "operations"}> ({
      %44 = "pdl.attribute"() : () -> !pdl.attribute
      %45 = "pdl.type"() : () -> !pdl.type
      %46 = "pdl.operation"(%44, %45) <{attributeValueNames = ["attr"], operandSegmentSizes = array<i32: 0, 1, 1>}> : (!pdl.attribute, !pdl.type) -> !pdl.operation
      %47 = "pdl.result"(%46) <{index = 0 : i32}> : (!pdl.operation) -> !pdl.value
      %48 = "pdl.operand"() : () -> !pdl.value
      %49 = "pdl.operation"(%47, %48) <{attributeValueNames = [], operandSegmentSizes = array<i32: 2, 0, 0>}> : (!pdl.value, !pdl.value) -> !pdl.operation
      "pdl.rewrite"(%49) <{name = "rewriter", operandSegmentSizes = array<i32: 1, 0>}> ({
      }) : (!pdl.operation) -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() ({
    "pdl.pattern"() <{benefit = 1 : i16, sym_name = "rewrite_with_args"}> ({
      %42 = "pdl.operand"() : () -> !pdl.value
      %43 = "pdl.operation"(%42) <{attributeValueNames = [], operandSegmentSizes = array<i32: 1, 0, 0>}> : (!pdl.value) -> !pdl.operation
      "pdl.rewrite"(%43, %42) <{name = "rewriter", operandSegmentSizes = array<i32: 1, 1>}> ({
      }) : (!pdl.operation, !pdl.value) -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() ({
    "pdl.pattern"() <{benefit = 2 : i16, sym_name = "rewrite_multi_root_optimal"}> ({
      %33 = "pdl.operand"() : () -> !pdl.value
      %34 = "pdl.operand"() : () -> !pdl.value
      %35 = "pdl.type"() : () -> !pdl.type
      %36 = "pdl.operation"(%33, %35) <{attributeValueNames = [], operandSegmentSizes = array<i32: 1, 0, 1>}> : (!pdl.value, !pdl.type) -> !pdl.operation
      %37 = "pdl.result"(%36) <{index = 0 : i32}> : (!pdl.operation) -> !pdl.value
      %38 = "pdl.operation"(%37) <{attributeValueNames = [], operandSegmentSizes = array<i32: 1, 0, 0>}> : (!pdl.value) -> !pdl.operation
      %39 = "pdl.operation"(%34, %35) <{attributeValueNames = [], operandSegmentSizes = array<i32: 1, 0, 1>}> : (!pdl.value, !pdl.type) -> !pdl.operation
      %40 = "pdl.result"(%39) <{index = 0 : i32}> : (!pdl.operation) -> !pdl.value
      %41 = "pdl.operation"(%37, %40) <{attributeValueNames = [], operandSegmentSizes = array<i32: 2, 0, 0>}> : (!pdl.value, !pdl.value) -> !pdl.operation
      "pdl.rewrite"(%38, %41) <{name = "rewriter", operandSegmentSizes = array<i32: 0, 2>}> ({
      }) : (!pdl.operation, !pdl.operation) -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() ({
    "pdl.pattern"() <{benefit = 2 : i16, sym_name = "rewrite_multi_root_forced"}> ({
      %24 = "pdl.operand"() : () -> !pdl.value
      %25 = "pdl.operand"() : () -> !pdl.value
      %26 = "pdl.type"() : () -> !pdl.type
      %27 = "pdl.operation"(%24, %26) <{attributeValueNames = [], operandSegmentSizes = array<i32: 1, 0, 1>}> : (!pdl.value, !pdl.type) -> !pdl.operation
      %28 = "pdl.result"(%27) <{index = 0 : i32}> : (!pdl.operation) -> !pdl.value
      %29 = "pdl.operation"(%28) <{attributeValueNames = [], operandSegmentSizes = array<i32: 1, 0, 0>}> : (!pdl.value) -> !pdl.operation
      %30 = "pdl.operation"(%25, %26) <{attributeValueNames = [], operandSegmentSizes = array<i32: 1, 0, 1>}> : (!pdl.value, !pdl.type) -> !pdl.operation
      %31 = "pdl.result"(%30) <{index = 0 : i32}> : (!pdl.operation) -> !pdl.value
      %32 = "pdl.operation"(%28, %31) <{attributeValueNames = [], operandSegmentSizes = array<i32: 2, 0, 0>}> : (!pdl.value, !pdl.value) -> !pdl.operation
      "pdl.rewrite"(%29, %32) <{name = "rewriter", operandSegmentSizes = array<i32: 1, 1>}> ({
      }) : (!pdl.operation, !pdl.operation) -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() ({
    "pdl.pattern"() <{benefit = 1 : i16, sym_name = "infer_type_from_operation_replace"}> ({
      %19 = "pdl.type"() <{constantType = i32}> : () -> !pdl.type
      %20 = "pdl.type"() : () -> !pdl.type
      %21 = "pdl.operation"(%19, %20) <{attributeValueNames = [], operandSegmentSizes = array<i32: 0, 0, 2>}> : (!pdl.type, !pdl.type) -> !pdl.operation
      "pdl.rewrite"(%21) <{operandSegmentSizes = array<i32: 1, 0>}> ({
        %22 = "pdl.type"() : () -> !pdl.type
        %23 = "pdl.operation"(%19, %22) <{attributeValueNames = [], opName = "foo.op", operandSegmentSizes = array<i32: 0, 0, 2>}> : (!pdl.type, !pdl.type) -> !pdl.operation
        "pdl.replace"(%21, %23) <{operandSegmentSizes = array<i32: 1, 1, 0>}> : (!pdl.operation, !pdl.operation) -> ()
      }) : (!pdl.operation) -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() ({
    "pdl.pattern"() <{benefit = 1 : i16, sym_name = "infer_type_from_type_used_in_match"}> ({
      %15 = "pdl.type"() <{constantType = i32}> : () -> !pdl.type
      %16 = "pdl.type"() : () -> !pdl.type
      %17 = "pdl.operation"(%15, %16) <{attributeValueNames = [], operandSegmentSizes = array<i32: 0, 0, 2>}> : (!pdl.type, !pdl.type) -> !pdl.operation
      "pdl.rewrite"(%17) <{operandSegmentSizes = array<i32: 1, 0>}> ({
        %18 = "pdl.operation"(%15, %16) <{attributeValueNames = [], opName = "foo.op", operandSegmentSizes = array<i32: 0, 0, 2>}> : (!pdl.type, !pdl.type) -> !pdl.operation
      }) : (!pdl.operation) -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() ({
    "pdl.pattern"() <{benefit = 1 : i16, sym_name = "infer_type_from_type_used_in_match"}> ({
      %11 = "pdl.types"() : () -> !pdl.range<type>
      %12 = "pdl.operation"(%11) <{attributeValueNames = [], operandSegmentSizes = array<i32: 0, 0, 1>}> : (!pdl.range<type>) -> !pdl.operation
      "pdl.rewrite"(%12) <{operandSegmentSizes = array<i32: 1, 0>}> ({
        %13 = "pdl.types"() <{constantTypes = [i32, i64]}> : () -> !pdl.range<type>
        %14 = "pdl.operation"(%11, %13) <{attributeValueNames = [], opName = "foo.op", operandSegmentSizes = array<i32: 0, 0, 2>}> : (!pdl.range<type>, !pdl.range<type>) -> !pdl.operation
      }) : (!pdl.operation) -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() ({
    "pdl.pattern"() <{benefit = 1 : i16, sym_name = "infer_type_from_type_used_in_match"}> ({
      %5 = "pdl.type"() : () -> !pdl.type
      %6 = "pdl.type"() : () -> !pdl.type
      %7 = "pdl.operand"(%5) : (!pdl.type) -> !pdl.value
      %8 = "pdl.operand"(%6) : (!pdl.type) -> !pdl.value
      %9 = "pdl.operation"(%7, %8) <{attributeValueNames = [], operandSegmentSizes = array<i32: 2, 0, 0>}> : (!pdl.value, !pdl.value) -> !pdl.operation
      "pdl.rewrite"(%9) <{operandSegmentSizes = array<i32: 1, 0>}> ({
        %10 = "pdl.operation"(%5, %6) <{attributeValueNames = [], opName = "foo.op", operandSegmentSizes = array<i32: 0, 0, 2>}> : (!pdl.type, !pdl.type) -> !pdl.operation
      }) : (!pdl.operation) -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() ({
    "pdl.pattern"() <{benefit = 1 : i16, sym_name = "infer_type_from_type_used_in_match"}> ({
      %1 = "pdl.types"() : () -> !pdl.range<type>
      %2 = "pdl.operands"(%1) : (!pdl.range<type>) -> !pdl.range<value>
      %3 = "pdl.operation"(%2) <{attributeValueNames = [], operandSegmentSizes = array<i32: 1, 0, 0>}> : (!pdl.range<value>) -> !pdl.operation
      "pdl.rewrite"(%3) <{operandSegmentSizes = array<i32: 1, 0>}> ({
        %4 = "pdl.operation"(%1) <{attributeValueNames = [], opName = "foo.op", operandSegmentSizes = array<i32: 0, 0, 1>}> : (!pdl.range<type>) -> !pdl.operation
      }) : (!pdl.operation) -> ()
    }) : () -> ()
  }) : () -> ()
  "builtin.module"() ({
    "pdl.pattern"() <{benefit = 1 : i16, sym_name = "apply_rewrite_with_no_results"}> ({
      %0 = "pdl.operation"() <{attributeValueNames = [], operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> !pdl.operation
      "pdl.rewrite"(%0) <{operandSegmentSizes = array<i32: 1, 0>}> ({
        "pdl.apply_native_rewrite"(%0) <{name = "NativeRewrite"}> : (!pdl.operation) -> ()
      }) : (!pdl.operation) -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()
