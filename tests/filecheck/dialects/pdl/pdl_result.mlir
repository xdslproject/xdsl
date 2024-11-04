// RUN: XDSL_ROUNDTRIP

pdl.pattern @extractResult : benefit(1) {
  %types = pdl.types
  %root = pdl.operation -> (%types : !pdl.range<type>)
  %result = pdl.result 1 of %root

  pdl.rewrite %root with "test_rewriter"
}

// CHECK: @extractResult
// CHECK: %{{.*}} = pdl.result 1 of %{{\S+}}

pdl.pattern @extractAllResults : benefit(1) {
  %types = pdl.types
  %root = pdl.operation -> (%types : !pdl.range<type>)
  %result = pdl.results of %root

  pdl.rewrite %root with "test_rewriter"
}

// CHECK: @extractAllResults
// CHECK: %{{.*}} = pdl.results of %{{\S+}}

pdl.pattern @extractOneResultRange : benefit(1) {
  %types = pdl.types
  %root = pdl.operation -> (%types : !pdl.range<type>)
  %result = pdl.results 1 of %root -> !pdl.range<value>

  pdl.rewrite %root with "test_rewriter"
}

// CHECK: @extractOneResultRange
// CHECK: %{{.*}} = pdl.results 1 of %{{\S+}}
