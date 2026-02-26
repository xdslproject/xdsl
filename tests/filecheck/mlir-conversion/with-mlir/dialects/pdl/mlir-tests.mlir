// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

builtin.module {
  pdl.pattern @operations : benefit(1) {
    // Operation with attributes and results.
    %attribute = pdl.attribute
    %type = pdl.type
    %op0 = pdl.operation {"attr" = %attribute} -> (%type : !pdl.type)
    %op0_result = pdl.result 0 of %op0

    // Operation with input.
    %input = pdl.operand
    %root = pdl.operation(%op0_result, %input : !pdl.value, !pdl.value)
    pdl.rewrite %root with "rewriter"
  }
}

// CHECK:      pdl.pattern @operations : benefit(1) {
// CHECK-NEXT:   %{{\d+}} = pdl.attribute
// CHECK-NEXT:   %{{\d+}} = pdl.type
// CHECK-NEXT:   %{{\d+}} = pdl.operation {"attr" = %{{\d+}}} -> (%{{\d+}} : !pdl.type)
// CHECK-NEXT:   %{{\d+}} = pdl.result 0 of %{{\d+}}
// CHECK-NEXT:   %{{\d+}} = pdl.operand
// CHECK-NEXT:   %{{\d+}} = pdl.operation (%{{\d+}}, %{{\d+}} : !pdl.value, !pdl.value)
// CHECK-NEXT:   pdl.rewrite %{{\d+}} with "rewriter"
// CHECK-NEXT: }


// -----

builtin.module {
  pdl.pattern @rewrite_with_args : benefit(1) {
    %input = pdl.operand
    %root = pdl.operation(%input : !pdl.value)
    pdl.rewrite %root with "rewriter"(%input : !pdl.value)
  }
}

// CHECK:      pdl.pattern @rewrite_with_args : benefit(1) {
// CHECK-NEXT:   %{{\d+}} = pdl.operand
// CHECK-NEXT:   %{{\d+}} = pdl.operation (%{{\d+}} : !pdl.value)
// CHECK-NEXT:   pdl.rewrite %{{\d+}} with "rewriter"(%{{\d+}} : !pdl.value)
// CHECK-NEXT: }

// -----

builtin.module {
  pdl.pattern @rewrite_multi_root_optimal : benefit(2) {
    %input1 = pdl.operand
    %input2 = pdl.operand
    %type = pdl.type
    %op1 = pdl.operation(%input1 : !pdl.value) -> (%type : !pdl.type)
    %val1 = pdl.result 0 of %op1
    %root1 = pdl.operation(%val1 : !pdl.value)
    %op2 = pdl.operation(%input2 : !pdl.value) -> (%type : !pdl.type)
    %val2 = pdl.result 0 of %op2
    %root2 = pdl.operation(%val1, %val2 : !pdl.value, !pdl.value)
    pdl.rewrite with "rewriter"(%root1, %root2 : !pdl.operation, !pdl.operation)
  }
}

// CHECK:      pdl.pattern @rewrite_multi_root_optimal : benefit(2) {
// CHECK-NEXT:   %{{\d+}} = pdl.operand
// CHECK-NEXT:   %{{\d+}} = pdl.operand
// CHECK-NEXT:   %{{\d+}} = pdl.type
// CHECK-NEXT:   %{{\d+}} = pdl.operation (%{{\d+}} : !pdl.value) -> (%{{\d+}} : !pdl.type)
// CHECK-NEXT:   %{{\d+}} = pdl.result 0 of %{{\d+}}
// CHECK-NEXT:   %{{\d+}} = pdl.operation (%{{\d+}} : !pdl.value)
// CHECK-NEXT:   %{{\d+}} = pdl.operation (%{{\d+}} : !pdl.value) -> (%{{\d+}} : !pdl.type)
// CHECK-NEXT:   %{{\d+}} = pdl.result 0 of %{{\d+}}
// CHECK-NEXT:   %{{\d+}} = pdl.operation (%{{\d+}}, %{{\d+}} : !pdl.value, !pdl.value)
// CHECK-NEXT:   pdl.rewrite with "rewriter"(%{{\d+}}, %{{\d+}} : !pdl.operation, !pdl.operation)
// CHECK-NEXT: }


// -----

builtin.module {
  pdl.pattern @rewrite_multi_root_forced : benefit(2) {
    %input1 = pdl.operand
    %input2 = pdl.operand
    %type = pdl.type
    %op1 = pdl.operation(%input1 : !pdl.value) -> (%type : !pdl.type)
    %val1 = pdl.result 0 of %op1
    %root1 = pdl.operation(%val1 : !pdl.value)
    %op2 = pdl.operation(%input2 : !pdl.value) -> (%type : !pdl.type)
    %val2 = pdl.result 0 of %op2
    %root2 = pdl.operation(%val1, %val2 : !pdl.value, !pdl.value)
    pdl.rewrite %root1 with "rewriter"(%root2 : !pdl.operation)
  }
}

// CHECK:      pdl.pattern @rewrite_multi_root_forced : benefit(2) {
// CHECK-NEXT:   %{{\d+}} = pdl.operand
// CHECK-NEXT:   %{{\d+}} = pdl.operand
// CHECK-NEXT:   %{{\d+}} = pdl.type
// CHECK-NEXT:   %{{\d+}} = pdl.operation (%{{\d+}} : !pdl.value) -> (%{{\d+}} : !pdl.type)
// CHECK-NEXT:   %{{\d+}} = pdl.result 0 of %{{\d+}}
// CHECK-NEXT:   %{{\d+}} = pdl.operation (%{{\d+}} : !pdl.value)
// CHECK-NEXT:   %{{\d+}} = pdl.operation (%{{\d+}} : !pdl.value) -> (%{{\d+}} : !pdl.type)
// CHECK-NEXT:   %{{\d+}} = pdl.result 0 of %{{\d+}}
// CHECK-NEXT:   %{{\d+}} = pdl.operation (%{{\d+}}, %{{\d+}} : !pdl.value, !pdl.value)
// CHECK-NEXT:   pdl.rewrite %{{\d+}} with "rewriter"(%{{\d+}} : !pdl.operation)
// CHECK-NEXT: }


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

// CHECK:      pdl.pattern @infer_type_from_operation_replace : benefit(1) {
// CHECK-NEXT:   %{{\d+}} = pdl.type : i32
// CHECK-NEXT:   %{{\d+}} = pdl.type
// CHECK-NEXT:   %{{\d+}} = pdl.operation -> (%{{\d+}}, %{{\d+}} : !pdl.type, !pdl.type)
// CHECK-NEXT:   pdl.rewrite %{{\d+}} {
// CHECK-NEXT:     %{{\d+}} = pdl.type
// CHECK-NEXT:     %{{\d+}} = pdl.operation "foo.op" -> (%{{\d+}}, %{{\d+}} : !pdl.type, !pdl.type)
// CHECK-NEXT:     pdl.replace %{{\d+}} with %{{\d+}}
// CHECK-NEXT:   }
// CHECK-NEXT: }


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

// CHECK:      pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
// CHECK-NEXT:   %{{\d+}} = pdl.type : i32
// CHECK-NEXT:   %{{\d+}} = pdl.type
// CHECK-NEXT:   %{{\d+}} = pdl.operation -> (%{{\d+}}, %{{\d+}} : !pdl.type, !pdl.type)
// CHECK-NEXT:   pdl.rewrite %{{\d+}} {
// CHECK-NEXT:     %{{\d+}} = pdl.operation "foo.op" -> (%{{\d+}}, %{{\d+}} : !pdl.type, !pdl.type)
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// Check that the result type of an operation within a rewrite can be inferred
// from the result types of an operation within the match block.
builtin.module {
  pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
    %types = pdl.types
    %root = pdl.operation -> (%types : !pdl.range<type>)
    pdl.rewrite %root {
      %otherTypes = pdl.types : [i32, i64]
      %newOp = pdl.operation "foo.op" -> (%types, %otherTypes : !pdl.range<type>, !pdl.range<type>)
    }
  }
}

// CHECK:      pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
// CHECK-NEXT:   %{{\d+}} = pdl.types
// CHECK-NEXT:   %{{\d+}} = pdl.operation -> (%{{\d+}} : !pdl.range<type>)
// CHECK-NEXT:   pdl.rewrite %{{\d+}} {
// CHECK-NEXT:     %{{\d+}} = pdl.types : [i32, i64]
// CHECK-NEXT:     %{{\d+}} = pdl.operation "foo.op" -> (%{{\d+}}, %{{\d+}} : !pdl.range<type>, !pdl.range<type>)
// CHECK-NEXT:   }
// CHECK-NEXT: }


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

// CHECK:      pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
// CHECK-NEXT:   %{{\d+}} = pdl.type
// CHECK-NEXT:   %{{\d+}} = pdl.type
// CHECK-NEXT:   %{{\d+}} = pdl.operand : %{{\d+}}
// CHECK-NEXT:   %{{\d+}} = pdl.operand : %{{\d+}}
// CHECK-NEXT:   %{{\d+}} = pdl.operation (%{{\d+}}, %{{\d+}} : !pdl.value, !pdl.value)
// CHECK-NEXT:   pdl.rewrite %{{\d+}} {
// CHECK-NEXT:     %{{\d+}} = pdl.operation "foo.op" -> (%{{\d+}}, %{{\d+}} : !pdl.type, !pdl.type)
// CHECK-NEXT:   }
// CHECK-NEXT: }


// -----

// Check that the result type of an operation within a rewrite can be inferred
// from the types of operands within the match block.
builtin.module {
  pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
    %types = pdl.types
    %operands = pdl.operands : %types
    %root = pdl.operation (%operands : !pdl.range<value>)
    pdl.rewrite %root {
      %newOp = pdl.operation "foo.op" -> (%types : !pdl.range<type>)
    }
  }
}

// CHECK:       pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
// CHECK-NEXT:    %{{\d+}} = pdl.types
// CHECK-NEXT:    %{{\d+}} = pdl.operands : %{{\d+}}
// CHECK-NEXT:    %{{\d+}} = pdl.operation (%{{\d+}} : !pdl.range<value>)
// CHECK-NEXT:    pdl.rewrite %{{\d+}} {
// CHECK-NEXT:      %{{\d+}} = pdl.operation "foo.op" -> (%{{\d+}} : !pdl.range<type>)
// CHECK-NEXT:    }
// CHECK-NEXT:  }


// -----

builtin.module {
  pdl.pattern @apply_rewrite_with_no_results : benefit(1) {
    %root = pdl.operation
    pdl.rewrite %root {
      pdl.apply_native_rewrite "NativeRewrite"(%root : !pdl.operation)
    }
  }
}

// CHECK:      pdl.pattern @apply_rewrite_with_no_results : benefit(1) {
// CHECK-NEXT:   %{{\d+}} = pdl.operation
// CHECK-NEXT:   pdl.rewrite %{{\d+}} {
// CHECK-NEXT:     pdl.apply_native_rewrite "NativeRewrite"(%{{\d+}} : !pdl.operation)
// CHECK-NEXT:   }
// CHECK-NEXT: }
