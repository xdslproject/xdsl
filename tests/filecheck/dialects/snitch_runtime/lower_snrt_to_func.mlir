// RUN: xdsl-opt -p lower-snrt-to-func %s | filecheck %s
"builtin.module"() ({
    "func.func"() ({
        // Runtime Info Getters
        %cluster_num = "snrt.cluster_num"() : () -> !riscv.reg<>
        // CHECK: %cluster_num = riscv_func.call @snrt_cluster_num() : () -> !riscv.reg<a0>

        // Barriers
        "snrt.cluster_hw_barrier"() : () -> ()
        // CHECK: riscv_func.call @snrt_cluster_hw_barrier() : () -> ()
        // DMA functions
        "snrt.dma_wait_all"() : () -> ()
        // CHECK: riscv_func.call @snrt_dma_wait_all() : () -> ()

        %dst_64 = riscv.li 100 : () -> !riscv.reg<>
        %src_64 = riscv.li 0 : () -> !riscv.reg<>
        %size = riscv.li 100 : () -> !riscv.reg<>
        %transfer_id = "snrt.dma_start_1d_wideptr"(%dst_64, %src_64, %size) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        // CHECK: %transfer_id{{.*}} = riscv_func.call @snrt_dma_start_1d_wideptr(%{{.*}}, %{{.*}}, %{{.*}}) : (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>) -> !riscv.reg<a0>

        %dst_32 = riscv.li 100 : () -> !riscv.reg<>
        %src_32 = riscv.li 0 : () -> !riscv.reg<>
        %size_2 = riscv.li 100 : () -> !riscv.reg<>
        %transfer_id_2 = "snrt.dma_start_1d"(%dst_32, %src_32, %size_2) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        // CHECK: %transfer_id_2{{.*}} = riscv_func.call @snrt_dma_start_1d(%{{.*}}, %{{.*}}, %{{.*}}) : (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>) -> !riscv.reg<a0>
        %repeat = riscv.li 1 : () -> !riscv.reg<>
        %src_stride = riscv.li 1 : () -> !riscv.reg<>
        %dst_stride = riscv.li 1 : () -> !riscv.reg<>
        %transfer_id_3 = "snrt.dma_start_2d_wideptr"(%dst_64, %src_64, %dst_stride, %src_stride, %size_2, %repeat) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        // CHECK: %transfer_id_3{{.*}} = riscv_func.call @snrt_dma_start_2d_wideptr(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.reg<a3>, !riscv.reg<a4>, !riscv.reg<a5>) -> !riscv.reg<a0>
        %transfer_id_4 = "snrt.dma_start_2d"(%dst_32, %src_32, %dst_stride, %src_stride, %size_2, %repeat) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        // CHECK: %transfer_id_4{{.*}} = riscv_func.call @snrt_dma_start_2d(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.reg<a3>, !riscv.reg<a4>, !riscv.reg<a5>) -> !riscv.reg<a0>
        "func.return"() : () -> ()
    }) {"sym_name" = "main", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
    // CHECK: riscv_func.func @snrt_cluster_num() -> !riscv.reg<a0>
    // CHECK: riscv_func.func @snrt_cluster_hw_barrier() -> ()
    // CHECK: riscv_func.func @snrt_dma_wait_all() -> ()
    // CHECK: riscv_func.func @snrt_dma_start_1d_wideptr(!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>) -> !riscv.reg<a0>
    // CHECK: riscv_func.func @snrt_dma_start_1d(!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>) -> !riscv.reg<a0>
    // CHECK: riscv_func.func @snrt_dma_start_2d_wideptr(!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.reg<a3>, !riscv.reg<a4>, !riscv.reg<a5>) -> !riscv.reg<a0>
    // CHECK: riscv_func.func @snrt_dma_start_2d(!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.reg<a3>, !riscv.reg<a4>, !riscv.reg<a5>) -> !riscv.reg<a0>
}) : () -> ()
