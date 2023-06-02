// RUN: xdsl-opt -p lower-snrt-to-func %s | filecheck %s
"builtin.module"() ({
    "func.func"() ({
        // Runtime Info Getters
        %cluster_num = "snrt.cluster_num"() : () -> ui32
        // CHECK: %cluster_num = "func.call"() {"callee" = @snrt_cluster_num} : () -> i32

        // Barriers
        "snrt.cluster_hw_barrier"() : () -> ()
        // CHECK: "func.call"() {"callee" = @snrt_cluster_hw_barrier} : () -> ()
    }) {"sym_name" = "main", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
    // CHECK: func.func private @snrt_cluster_num() -> i32
    // CHECK: func.func private @snrt_cluster_hw_barrier() -> ()
}) : () -> ()
