// RUN: XDSL_AUTO_ROUNDTRIP

builtin.module {
  pdl.pattern @operations : benefit(1) {
    // Operation with attributes and results.
    %attribute = pdl.attribute
    %type = pdl.type
    %op0 = pdl.operation {"attr" = %attribute} -> (%type : !pdl.type)
    %op0_result = pdl.result 0 of %op0

    // Operation with input.
    %input = pdl.operand
    %root = pdl.operation (%op0_result, %input : !pdl.value, !pdl.value)
    pdl.rewrite %root with "rewriter"
  }
}


// -----

builtin.module {
  pdl.pattern @rewrite_with_args : benefit(1) {
    %input = pdl.operand
    %root = pdl.operation (%input : !pdl.value)
    pdl.rewrite %root with "rewriter"(%input : !pdl.value)
  }
}

// -----

builtin.module {
  pdl.pattern @rewrite_multi_root_optimal : benefit(2) {
    %input1 = pdl.operand
    %input2 = pdl.operand
    %type = pdl.type
    %op1 = pdl.operation (%input1 : !pdl.value) -> (%type : !pdl.type)
    %val1 = pdl.result 0 of %op1
    %root1 = pdl.operation (%val1 : !pdl.value)
    %op2 = pdl.operation (%input2 : !pdl.value) -> (%type : !pdl.type)
    %val2 = pdl.result 0 of %op2
    %root2 = pdl.operation (%val1, %val2 : !pdl.value, !pdl.value)
    pdl.rewrite with "rewriter"(%root1, %root2 : !pdl.operation, !pdl.operation)
  }
}


// -----

builtin.module {
  pdl.pattern @rewrite_multi_root_forced : benefit(2) {
    %input1 = pdl.operand
    %input2 = pdl.operand
    %type = pdl.type
    %op1 = pdl.operation (%input1 : !pdl.value) -> (%type : !pdl.type)
    %val1 = pdl.result 0 of %op1
    %root1 = pdl.operation (%val1 : !pdl.value)
    %op2 = pdl.operation (%input2 : !pdl.value) -> (%type : !pdl.type)
    %val2 = pdl.result 0 of %op2
    %root2 = pdl.operation (%val1, %val2 : !pdl.value, !pdl.value)
    pdl.rewrite %root1 with "rewriter"(%root2 : !pdl.operation)
  }
}


// -----

// Check that the result type of an operation within a rewrite can be inferred
// from a pdl.replace.
builtin.module {
  pdl.pattern @infer_type_from_operation_replace : benefit(1) {
    %type1 = pdl.type : i32
    %type2 = pdl.type
    %root = pdl.operation -> (%type1, %type2 : !pdl.type, !pdl.type)
    pdl.rewrite %root {
      %type3 = pdl.type
      %newOp = pdl.operation "foo.op" -> (%type1, %type3 : !pdl.type, !pdl.type)
      pdl.replace %root with %newOp
    }
  }
}


// -----

// Check that the result type of an operation within a rewrite can be inferred
// from the result types of an operation within the match block.
builtin.module {
  pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
    %type1 = pdl.type : i32
    %type2 = pdl.type
    %root = pdl.operation -> (%type1, %type2 : !pdl.type, !pdl.type)
    pdl.rewrite %root {
      %newOp = pdl.operation "foo.op" -> (%type1, %type2 : !pdl.type, !pdl.type)
    }
  }
}

// -----

// Check that the result type of an operation within a rewrite can be inferred
// from the result types of an operation within the match block.
builtin.module {
  pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
    %types = pdl.types
    %root = pdl.operation -> (%types : !pdl.range<!pdl.type>)
    pdl.rewrite %root {
      %otherTypes = pdl.types : [i32, i64]
      %newOp = pdl.operation "foo.op" -> (%types, %otherTypes : !pdl.range<!pdl.type>, !pdl.range<!pdl.type>)
    }
  }
}


// -----

// Check that the result type of an operation within a rewrite can be inferred
// from the type of an operand within the match block.
builtin.module {
  pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
    %type1 = pdl.type
    %type2 = pdl.type
    %operand1 = pdl.operand : %type1
    %operand2 = pdl.operand : %type2
    %root = pdl.operation (%operand1, %operand2 : !pdl.value, !pdl.value)
    pdl.rewrite %root {
      %newOp = pdl.operation "foo.op" -> (%type1, %type2 : !pdl.type, !pdl.type)
    }
  }
}


// -----

// Check that the result type of an operation within a rewrite can be inferred
// from the types of operands within the match block.
builtin.module {
  pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
    %types = pdl.types
    %operands = pdl.operands : %types
    %root = pdl.operation (%operands : !pdl.range<!pdl.value>)
    pdl.rewrite %root {
      %newOp = pdl.operation "foo.op" -> (%types : !pdl.range<!pdl.type>)
    }
  }
}


// -----

builtin.module {
  pdl.pattern @apply_rewrite_with_no_results : benefit(1) {
    %root = pdl.operation
    pdl.rewrite %root {
      pdl.apply_native_rewrite "NativeRewrite"(%root : !pdl.operation)
    }
  }
}
