// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"()({
    "gpu.module"()({
        %n = "arith.constant"() {"value" = 13 : index} : () -> index
        "gpu.module_end"() : () -> ()
        %one = "arith.constant"() {"value" = 1 : index} : () -> index
    }) {"sym_name" = "gpu"} : () -> ()
}) {} : () -> ()

// CHECK: 'gpu.module_end' must be the last operation in its parent block
