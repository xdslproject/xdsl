// RUN: XDSL_ROUNDTRIP

%source, %dest = "test.op"() : () -> (index, index)

aie.flow(%source, %dest) <!aie.wire_bundle<"source">: 1, !aie.wire_bundle<"dest">: 2>

// CHECK: aie.flow
