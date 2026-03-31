// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

module{
  %0 = arith.constant true
  cf.assert %0, "some message"
}

// CHECK: cf.assert %{{.*}}, "some message"
// CHECK-GENERIC: "cf.assert"(%{{.*}}) <{msg = "some message"}> : (i1) -> ()
