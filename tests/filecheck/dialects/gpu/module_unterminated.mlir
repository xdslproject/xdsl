// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"()({
    "gpu.module"()({
    ^0:
      "test.termop"() : () -> ()
    }) {"sym_name" = "gpu"} : () -> ()
}) {} : () -> ()

// CHECK: gpu.module must end with gpu.module_end
