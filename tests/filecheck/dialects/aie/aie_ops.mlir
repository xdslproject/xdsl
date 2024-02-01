// RUN: XDSL_ROUNDTRIP

%source, %dest = "test.op"() : () -> (index, index)

aie.flow(%source, %dest) <#aie.wire_bundle<"source"> : 1 : i64, #aie.wire_bundle<"dest"> : 2 : i64>

// CHECK: aie.flow(%source, %dest) <#aie.wire_bundle<"source"> : 1 : i64, #aie.wire_bundle<"dest"> : 2 : i64>
