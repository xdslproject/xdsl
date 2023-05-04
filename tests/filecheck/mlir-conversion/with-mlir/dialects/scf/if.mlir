// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() {"value" = false} : () -> i1
  "scf.if"(%0) ({
    "scf.yield"() : () -> ()
  }, {
    "scf.yield"() : () -> ()
  }) : (i1) -> ()
}) : () -> ()

// CHECK:        "scf.if"(%{{\d+}}) ({
// CHECK-NEXT:     "scf.yield"() : () -> ()
// CHECK-NEXT:   }, {
// CHECK-NEXT:     "scf.yield"() : () -> ()
// CHECK-NEXT:   }) : (i1) -> ()
