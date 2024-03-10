// RUN: XDSL_ROUNDTRIP

pdl.pattern @emptyRanges : benefit(1) {
// CHECK: @emptyRanges
  %types = pdl.range : !pdl.range<type>
// CHECK: %{{.*}} = pdl.range : !pdl.range<type>
  %values = pdl.range : !pdl.range<value>
// CHECK: %{{.*}} = pdl.range : !pdl.range<value>

  %root = pdl.operation (%values: !pdl.range<value>) -> (%types: !pdl.range<type>)
  pdl.rewrite %root with "test_rewriter"
}


pdl.pattern @nonEmptyRanges : benefit(1) {
// CHECK: @nonEmptyRanges
  %type1 = pdl.type : !pdl.type
  %type1_range = pdl.range %type1 : !pdl.type
// CHECK: %{{.*}} = pdl.range %{{.*}} : !pdl.type
  %type2 = pdl.type : !pdl.type
  %types = pdl.range %type1_range, %type2 : !pdl.range<type>, !pdl.type
// CHECK: %{{.*}} = pdl.range %{{.*}}, %{{.*}} : !pdl.range<type>, !pdl.type

  %value1 = pdl.operand : %type1
  %value1_range = pdl.range %value1 : !pdl.value
// CHECK: %{{.*}} = pdl.range %{{.*}} : !pdl.value
  %value2 = pdl.operand : %type2
  %values = pdl.range %value1_range, %value2 : !pdl.range<value>, !pdl.value
// CHECK: %{{.*}} = pdl.range %{{.*}}, %{{.*}} : !pdl.range<value>, !pdl.value

  %root = pdl.operation (%values: !pdl.range<value>) -> (%types: !pdl.range<type>)
  pdl.rewrite %root with "test_rewriter"
}



