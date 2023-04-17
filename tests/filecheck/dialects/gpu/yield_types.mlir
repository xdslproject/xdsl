// RUN: xdsl-opt -t mlir %s --verify-diagnostics | filecheck %s

"builtin.module"()({
    "gpu.module"()({
        %init = "arith.constant"() {"value" = 42 : index} : () -> index
        %sum = "gpu.all_reduce"(%init) ({
        ^bb(%lhs : index, %rhs : index):
            %sum = "arith.addi"(%lhs, %rhs) : (index, index) -> index
            %float = "arith.constant"() {"value" = 42.0 : f32} : () -> f32
            "gpu.yield"(%float) : (f32) -> ()
        }) {} : (index) -> index
        "gpu.module_end"() : () -> ()
    }) {"sym_name" = "gpu"} : () -> ()
}) {} : () -> ()

// CHECK: Expected ['index'], got ['f32']. The gpu.yield values types must match its enclosing operation result types.
