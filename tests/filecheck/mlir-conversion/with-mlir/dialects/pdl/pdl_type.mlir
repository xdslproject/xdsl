// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

pdl.pattern @unboundedType : benefit(1) {
  // An unbounded type
  %type = pdl.type

  %root = pdl.operation -> (%type : !pdl.type)
  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @unboundedType
// CHECK: %{{\d+}} = pdl.type

pdl.pattern @knownType : benefit(1) {
  // A known type
  %type = pdl.type : i32

  %root = pdl.operation -> (%type : !pdl.type)
  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @knownType
// CHECK: %{{\d+}} = pdl.type : i32

pdl.pattern @unboundedTypes : benefit(1) {
  // Unbounded types
  %type = pdl.types

  %root = pdl.operation -> (%type : !pdl.range<type>)
  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @unboundedTypes
// CHECK: %{{\d+}} = pdl.types

pdl.pattern @knownTypes : benefit(1) {
  // Known types
  %type = pdl.types : [i32, i64]

  %root = pdl.operation -> (%type : !pdl.range<type>)
  pdl.rewrite %root with "test_rewriter"(%root : !pdl.operation)
}

// CHECK: @knownTypes
// CHECK: %{{\d+}} = pdl.types : [i32, i64]
