// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

pdl.pattern @extractResult : benefit(1) {
  %types = pdl.types
  %root = pdl.operation -> (%types : !pdl.range<type>)
  %result = pdl.result 1 of %root

  pdl.rewrite %root with "test_rewriter"
}

// CHECK: @extractResult
// CHECK: %{{\d+}} = pdl.result 1 of %{{\d+}}

pdl.pattern @extractAllResults : benefit(1) {
  %types = pdl.types
  %root = pdl.operation -> (%types : !pdl.range<type>)
  %result = pdl.results of %root

  pdl.rewrite %root with "test_rewriter"
}

// CHECK: @extractAllResults
// CHECK: %{{\d+}} = pdl.results of %{{\d+}}

pdl.pattern @extractOneResultRange : benefit(1) {
  %types = pdl.types
  %root = pdl.operation -> (%types : !pdl.range<type>)
  %result = pdl.results 1 of %root -> !pdl.range<value>

  pdl.rewrite %root with "test_rewriter"
}

// CHECK: @extractOneResultRange
// CHECK: %{{\d+}} = pdl.results 1 of %{{\d+}}
