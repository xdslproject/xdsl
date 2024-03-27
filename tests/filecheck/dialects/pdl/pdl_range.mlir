// RUN: XDSL_ROUNDTRIP

// CHECK: "test.op"() : () -> (!pdl.range<operation>, !pdl.range<attribute>)
"test.op"() : () -> (!pdl.range<operation>, !pdl.range<attribute>)

pdl.pattern @emptyRanges : benefit(1) {
// CHECK: @emptyRanges
  %root = pdl.operation
  pdl.rewrite %root {
    %types = pdl.range : !pdl.range<type>
  // CHECK: %{{.*}} = pdl.range : !pdl.range<type>
    %values = pdl.range : !pdl.range<value>
  // CHECK: %{{.*}} = pdl.range : !pdl.range<value>

    %rep = pdl.operation "test.op" (%values: !pdl.range<value>) -> (%types: !pdl.range<type>)
  }
}


pdl.pattern @nonEmptyRanges : benefit(1) {
// CHECK: @nonEmptyRanges
  %type1 = pdl.type : !pdl.type
  %type2 = pdl.type : !pdl.type
  %value1 = pdl.operand : %type1
  %value2 = pdl.operand : %type2
  
  %root = pdl.operation (%value1, %value2: !pdl.value, !pdl.value)
  pdl.rewrite %root {
    %type1_range = pdl.range %type1 : !pdl.type
  // CHECK: %{{.*}} = pdl.range %{{.*}} : !pdl.type
    %types = pdl.range %type1_range, %type2 : !pdl.range<type>, !pdl.type
  // CHECK: %{{.*}} = pdl.range %{{.*}}, %{{.*}} : !pdl.range<type>, !pdl.type

    %value1_range = pdl.range %value1 : !pdl.value
  // CHECK: %{{.*}} = pdl.range %{{.*}} : !pdl.value
    %values = pdl.range %value1_range, %value2 : !pdl.range<value>, !pdl.value
  // CHECK: %{{.*}} = pdl.range %{{.*}}, %{{.*}} : !pdl.range<value>, !pdl.value

    %rep = pdl.operation "test.op" (%values: !pdl.range<value>) -> (%types: !pdl.range<type>)
  }
}
