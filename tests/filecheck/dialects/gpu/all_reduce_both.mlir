// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"()({
    "gpu.module"()({
        %init = "arith.constant"() {"value" = 42 : index} : () -> index
        %sum = "gpu.all_reduce"(%init) ({
        ^bb(%lhs : index, %rhs : index):
            %sum = "arith.addi"(%lhs, %rhs) : (index, index) -> index
            "gpu.yield"(%sum) : (index) -> ()
        }) {"op" = #gpu<all_reduce_op add>} : (index) -> index
        "gpu.module_end"() : () -> ()
    }) {"sym_name" = "gpu"} : () -> ()
}) {} : () -> ()

// CHECK: gpu.all_reduce can't have both a non-empty region and an op attribute.
