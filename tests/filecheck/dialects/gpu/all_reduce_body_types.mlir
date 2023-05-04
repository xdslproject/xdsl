// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"()({
    "gpu.module"()({
        %init = "arith.constant"() {"value" = 42 : index} : () -> index
        %sum = "gpu.all_reduce"(%init) ({
        ^bb(%lhs : index):
            %c = "arith.constant"() {"value" = 42 : index} : () -> index
            %sum = "arith.addi"(%lhs, %c) : (index, index) -> index
            "gpu.yield"(%sum) : (index) -> ()
        }) : (index) -> index
        "gpu.module_end"() : () -> ()
    }) {"sym_name" = "gpu"} : () -> ()
}) {} : () -> ()

// CHECK: Expected ['index', 'index'], got ['index']. A gpu.all_reduce's body must have two arguments matching the result type.
