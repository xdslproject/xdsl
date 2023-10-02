// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = true} : () -> i1
  "cf.assert"(%0) {msg = "some message"} : (i1) -> ()
}) : () -> ()

// CHECK: "cf.assert"(%{{.*}}) <{msg = "some message"}> : (i1) -> ()
