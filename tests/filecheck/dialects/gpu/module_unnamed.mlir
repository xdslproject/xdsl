// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"()({
    "gpu.module"()({
        "gpu.module_end"() : () -> ()
    }) {} : () -> ()
}) {} : () -> ()

// CHECK: attribute sym_name expected
