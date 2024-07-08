// RUN: XDSL_ROUNDTRIP

%0 = "test.op"() : () -> (!transform.affine_map)
%1 = "test.op"() : () -> (!transform.any_op)
%2 = "test.op"() : () -> (!transform.any_param)
%3 = "test.op"() : () -> (!transform.any_value)

// CHECK-NEXT:  module {
// CHECK-NEXT:    %0 = "test.op"() : () -> !transform.affine_map
// CHECK-NEXT:    %1 = "test.op"() : () -> !transform.any_op
// CHECK-NEXT:    %2 = "test.op"() : () -> !transform.any_param
// CHECK-NEXT:    %3 = "test.op"() : () -> !transform.any_value
// CHECK-NEXT:  }
