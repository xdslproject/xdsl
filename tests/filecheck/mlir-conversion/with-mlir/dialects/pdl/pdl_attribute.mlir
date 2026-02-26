// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

pdl.pattern @unboundedAttribute : benefit(1) {
  // An unbounded attribute
  %attribute = pdl.attribute

  %root = pdl.operation {"attr" = %attribute}
  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @unboundedAttribute
// CHECK: %{{\d+}} = pdl.attribute

pdl.pattern @typedAttribute : benefit(1) {
  %type = pdl.type : i32
  // A typed attribute
  %attribute = pdl.attribute : %type

  %root = pdl.operation {"attr" = %attribute}
  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @typedAttribute
// CHECK: %{{\d+}} = pdl.attribute : %{{\d+}}

pdl.pattern @constantAttribute : benefit(1) {
  // A constant attribute
  %attribute = pdl.attribute = 0 : i32

  %root = pdl.operation {"attr" = %attribute}
  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @constantAttribute
// CHECK: %{{\d+}} = pdl.attribute = 0 : i32
