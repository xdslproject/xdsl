// RUN: XDSL_ROUNDTRIP

"test.op"() {
    bigint = !bigint.bigint,
} : () -> ()

%bigint = "test.op"() : () -> (!bigint.bigint)

// CHECK:       builtin.module {
// CHECK-NEXT:    "test.op"() {bigint = !bigint.bigint} : () -> ()
// CHECK-NEXT:    %bigint = "test.op"() : () -> (!bigint.bigint)
// CHECK-NEXT:  }
