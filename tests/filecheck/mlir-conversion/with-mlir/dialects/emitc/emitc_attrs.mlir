// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

//===----------------------------------------------------------------------===//
// OpaqueAttr
//===----------------------------------------------------------------------===//

"test.op"() {
  // CHECK: opaque_attr = #emitc.opaque<"some_value">
  opaque_attr = #emitc.opaque<"some_value">,
  // CHECK-SAME: quoted_attr = #emitc.opaque<"\"quoted_attr\"">
  quoted_attr = #emitc.opaque<"\"quoted_attr\"">
}: () -> ()
