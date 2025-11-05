// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = true} : () -> i1
  "cf.assert"(%0) <{msg = "some message"}> : (i1) -> ()
}) : () -> ()

// CHECK: cf.assert %{{.*}}, "some message"
// CHECK-GENERIC: "cf.assert"(%{{.*}}) <{msg = "some message"}> : (i1) -> ()
