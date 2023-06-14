// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"()({
    "gpu.module"()({
    ^0:
    "gpu.module_end"() : () -> ()
    }) {"sym_name" = "gpu"} : () -> ()
    "gpu.module_end"() : () -> ()
}) {} : () -> ()

// CHECK: 'gpu.module_end' expects parent op 'gpu.module'
