// RUN: XDSL_ROUNDTRIP

%bigint = "test.op"() : () -> !bigint.bigint

// CHECK:       builtin.module {
// CHECK-NEXT:    %bigint = "test.op"() : () -> !bigint.bigint
// CHECK-NEXT:  }
