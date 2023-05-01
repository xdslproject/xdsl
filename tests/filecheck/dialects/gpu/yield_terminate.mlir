// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"()({
    "gpu.module"()({
        %init = "arith.constant"() {"value" = 42 : index} : () -> index
        %sum = "gpu.all_reduce"(%init) ({
        ^bb(%lhs : index, %rhs : index):
            %sum = "arith.addi"(%lhs, %rhs) : (index, index) -> index
            %float = "arith.constant"() {"value" = 42.0 : f32} : () -> f32
            "gpu.yield"(%float) : (f32) -> ()
            %more = "arith.constant"() {"value" = 84.0 : f32} : () -> f32
        }) {} : (index) -> index
        "gpu.module_end"() : () -> ()
    }) {"sym_name" = "gpu"} : () -> ()
}) {} : () -> ()

// CHECK: A gpu.yield must terminate its parent block
