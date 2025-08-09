// RUN: xdsl-opt %s | mlir-opt | xdsl-opt | filecheck %s
// RUN: xdsl-opt %s --print-op-generic | mlir-opt --mlir-print-op-generic | xdsl-opt | filecheck %s

//===----------------------------------------------------------------------===//
// OpaqueAttr
//===----------------------------------------------------------------------===//

"test.op"() {
  // CHECK: opaque_attr = #emitc.opaque<"some_value">
  opaque_attr = #emitc.opaque<"some_value">,
  // CHECK-SAME: quoted_attr = #emitc.opaque<"\"quoted_attr\"">
  quoted_attr = #emitc.opaque<"\"quoted_attr\"">
}: () -> ()
