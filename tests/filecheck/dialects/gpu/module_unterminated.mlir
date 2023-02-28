// RUN: xdsl-opt -t mlir %s --verify-diagnostics | filecheck %s

"builtin.module"()({
    "gpu.module"()({
    }) {"sym_name" = "gpu"} : () -> ()
}) {} : () -> ()

// CHECK: gpu.module must end with gpu.module_end
