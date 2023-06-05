// RUN: xdsl-opt -p lower-snrt-to-func %s | filecheck %s
"builtin.module"() ({
    "func.func"() ({
        // Runtime Info Getters
        %cluster_num = "snrt.cluster_num"() : () -> ui32
        // CHECK: %cluster_num = "func.call"() {"callee" = @snrt_cluster_num} : () -> i32

        // Barriers
        "snrt.cluster_hw_barrier"() : () -> ()
        // CHECK: "func.call"() {"callee" = @snrt_cluster_hw_barrier} : () -> ()
        // DMA functions
        "snrt.dma_wait_all"() : () -> ()

        %dst_32 = "arith.constant"() {"value" = 100: ui32} : () -> ui32
        %src_32 = "arith.constant"() {"value" = 0: ui32} : () -> ui32
        %size_2 = "arith.constant"() {"value" = 100: index} : () -> index
        %transfer_id_2 = "snrt.dma_start_1d"(%dst_32, %src_32, %size_2) : (ui32, ui32, index) -> ui32
        // CHECK: %transfer_id_2 = "func.call"(%dst_32, %src_32, %size_2) {"callee" = @snrt_dma_start_1d} : (ui32, ui32, index) -> ui32
        %repeat = "arith.constant"() {"value" = 1: index} : () -> index
        %src_stride = "arith.constant"() {"value" = 1: index} : () -> index
        %dst_stride = "arith.constant"() {"value" = 1: index} : () -> index
        %transfer_id_4 = "snrt.dma_start_2d"(%dst_32, %src_32, %dst_stride, %src_stride, %size_2, %repeat) : (ui32, ui32, index, index, index, index) -> ui32
        // CHECK: %transfer_id_4 = "func.call"(%dst_32, %src_32, %dst_stride, %src_stride, %size_2, %repeat) {"callee" = @snrt_dma_start_2d} : (ui32, ui32, index, index, index, index) -> ui32


    }) {"sym_name" = "main", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
    // CHECK: func.func private @snrt_cluster_num() -> i32
    // CHECK: func.func private @snrt_cluster_hw_barrier() -> ()
    // CHECK: func.func private @snrt_dma_wait_all() -> ()
    // CHECK: func.func private @snrt_dma_start_1d(ui32, ui32, index) -> ui32
    // CHECK: func.func private @snrt_dma_start_2d(ui32, ui32, index, index, index, index) -> ui32
}) : () -> ()
