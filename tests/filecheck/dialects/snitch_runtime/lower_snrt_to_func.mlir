// RUN: xdsl-opt -p lower-snrt-to-func %s | filecheck %s
"builtin.module"() ({
    "func.func"() ({
        // Runtime Info Getters
        %cluster_num = "snrt.cluster_num"() : () -> i32
        // CHECK: %cluster_num = func.call @snrt_cluster_num() : () -> i32

        // Barriers
        "snrt.cluster_hw_barrier"() : () -> ()
        // CHECK: func.call @snrt_cluster_hw_barrier() : () -> ()
        // DMA functions
        "snrt.dma_wait_all"() : () -> ()
        // CHECK: func.call @snrt_dma_wait_all() : () -> ()

        %dst_64 = "arith.constant"() {"value" = 100 : i64} : () -> i64
        %src_64 = "arith.constant"() {"value" = 0 : i64} : () -> i64
        %size = "arith.constant"() {"value" = 100 : index} : () -> index
        %transfer_id = "snrt.dma_start_1d_wideptr"(%dst_64, %src_64, %size) : (i64, i64, index) -> i32
        // CHECK: %transfer_id = func.call @snrt_dma_start_1d_wideptr(%dst_64, %src_64, %size) : (i64, i64, index) -> i32

        %dst_32 = "arith.constant"() {"value" = 100: i32} : () -> i32
        %src_32 = "arith.constant"() {"value" = 0: i32} : () -> i32
        %size_2 = "arith.constant"() {"value" = 100: index} : () -> index
        %transfer_id_2 = "snrt.dma_start_1d"(%dst_32, %src_32, %size_2) : (i32, i32, index) -> i32
        // CHECK: %transfer_id_2 = func.call @snrt_dma_start_1d(%dst_32, %src_32, %size_2) : (i32, i32, index) -> i32
        %repeat = "arith.constant"() {"value" = 1: index} : () -> index
        %src_stride = "arith.constant"() {"value" = 1: index} : () -> index
        %dst_stride = "arith.constant"() {"value" = 1: index} : () -> index
        %transfer_id_3 = "snrt.dma_start_2d_wideptr"(%dst_64, %src_64, %dst_stride, %src_stride, %size_2, %repeat) : (i64, i64, index, index, index, index) -> i32
        // CHECK: %transfer_id_3 = func.call @snrt_dma_start_2d_wideptr(%dst_64, %src_64, %dst_stride, %src_stride, %size_2, %repeat) : (i64, i64, index, index, index, index) -> i32
        %transfer_id_4 = "snrt.dma_start_2d"(%dst_32, %src_32, %dst_stride, %src_stride, %size_2, %repeat) : (i32, i32, index, index, index, index) -> i32
        // CHECK: %transfer_id_4 = func.call @snrt_dma_start_2d(%dst_32, %src_32, %dst_stride, %src_stride, %size_2, %repeat) : (i32, i32, index, index, index, index) -> i32
        "func.return"() : () -> ()
    }) {"sym_name" = "main", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
    // CHECK: func.func private @snrt_cluster_num() -> i32
    // CHECK: func.func private @snrt_cluster_hw_barrier() -> ()
    // CHECK: func.func private @snrt_dma_wait_all() -> ()
    // CHECK: func.func private @snrt_dma_start_1d_wideptr(i64, i64, index) -> i32
    // CHECK: func.func private @snrt_dma_start_1d(i32, i32, index) -> i32
    // CHECK: func.func private @snrt_dma_start_2d_wideptr(i64, i64, index, index, index, index) -> i32
    // CHECK: func.func private @snrt_dma_start_2d(i32, i32, index, index, index, index) -> i32
}) : () -> ()
