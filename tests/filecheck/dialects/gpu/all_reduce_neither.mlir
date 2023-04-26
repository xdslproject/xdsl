// RUN: xdsl-opt -t mlir %s --verify-diagnostics | filecheck %s

"builtin.module"()({
    "gpu.module"()({
        %init = "arith.constant"() {"value" = 42 : index} : () -> index
        %sum = "gpu.all_reduce"(%init) ({}) : (index) -> index
        "gpu.module_end"() : () -> ()
    }) {"sym_name" = "gpu"} : () -> ()
}) {} : () -> ()

// CHECK: gpu.all_reduce need either a non empty body or an op attribute.
