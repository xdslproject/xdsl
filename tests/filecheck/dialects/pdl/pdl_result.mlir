// RUN: XDSL_ROUNDTRIP

pdl.pattern @extractResult : benefit(1) {
  %types = pdl.types
  %root = pdl.operation -> (%types : !pdl.range<!pdl.type>)
  %result = pdl.result 1 of %root

  pdl.rewrite %root with "test_rewriter"
}

// CHECK: @extractResult
// CHECK: %{{.*}} = pdl.result 1 of %{{.*}}

pdl.pattern @extractAllResults : benefit(1) {
  %types = pdl.types
  %root = pdl.operation -> (%types : !pdl.range<!pdl.type>)
  %result = pdl.results of %root

  pdl.rewrite %root with "test_rewriter"
}

// CHECK: @extractAllResults
// CHECK: %{{.*}} = pdl.results of %{{.*}}

pdl.pattern @extractOneResultRange : benefit(1) {
  %types = pdl.types
  %root = pdl.operation -> (%types : !pdl.range<!pdl.type>)
  %result = pdl.results 1 of %root -> !pdl.range<!pdl.value>

  pdl.rewrite %root with "test_rewriter"
}

// CHECK: @extractOneResultRange
// CHECK: %{{.*}} = pdl.results 1 of %{{.*}}