// RUN: xdsl-opt -t mlir %s | mlir-opt --mlir-print-op-generic | xdsl-opt -f mlir -t mlir | filecheck %s

"builtin.module"() ({
    "gpu.module"()({
        "gpu.module_end"() : () -> ()
    }) {"sym_name" = "gpu"} : () -> ()
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:     "gpu.module"() ({
// CHECK-NEXT:          "gpu.module_end"() : () -> ()
// CHECK-NEXT:     }) {"sym_name" = "gpu"} : () -> ()

// CHECK-NEXT: }) : () -> ()