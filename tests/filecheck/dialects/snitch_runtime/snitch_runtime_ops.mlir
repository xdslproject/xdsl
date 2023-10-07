// RUN: XDSL_ROUNDTRIP
builtin.module {
  func.func @main() {
        // Barriers
        "snrt.cluster_hw_barrier"() : () -> ()
        // CHECK: "snrt.cluster_hw_barrier"() : () -> ()
        "snrt.cluster_sw_barrier"() : () -> ()
        // CHECK: "snrt.cluster_sw_barrier"() : () -> ()
        "snrt.global_barrier"() : () -> ()
        // CHECK: "snrt.global_barrier"() : () -> ()

        // Runtime Info Getters
        %global_core_base_hartid = "snrt.global_core_base_hartid"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.global_core_base_hartid"() : () -> !riscv.reg<>
        %global_core_idx = "snrt.global_core_idx"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.global_core_idx"() : () -> !riscv.reg<>
        %global_core_num = "snrt.global_core_num"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.global_core_num"() : () -> !riscv.reg<>
        %global_compute_core_idx = "snrt.global_compute_core_idx"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.global_compute_core_idx"() : () -> !riscv.reg<>
        %global_compute_core_num = "snrt.global_compute_core_num"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.global_compute_core_num"() : () -> !riscv.reg<>
        %global_dm_core_idx = "snrt.global_dm_core_idx"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.global_dm_core_idx"() : () -> !riscv.reg<>
        %global_dm_core_num = "snrt.global_dm_core_num"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.global_dm_core_num"() : () -> !riscv.reg<>
        %cluster_core_base_hartid = "snrt.cluster_core_base_hartid"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.cluster_core_base_hartid"() : () -> !riscv.reg<>
        %gcluster_core_idx = "snrt.cluster_core_idx"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.cluster_core_idx"() : () -> !riscv.reg<>
        %cluster_core_num = "snrt.cluster_core_num"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.cluster_core_num"() : () -> !riscv.reg<>
        %cluster_compute_core_idx = "snrt.cluster_compute_core_idx"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.cluster_compute_core_idx"() : () -> !riscv.reg<>
        %cluster_compute_core_num = "snrt.cluster_compute_core_num"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.cluster_compute_core_num"() : () -> !riscv.reg<>
        %cluster_dm_core_idx = "snrt.cluster_dm_core_idx"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.cluster_dm_core_idx"() : () -> !riscv.reg<>
        %cluster_dm_core_num = "snrt.cluster_dm_core_num"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.cluster_dm_core_num"() : () -> !riscv.reg<>
        %cluster_idx = "snrt.cluster_idx"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.cluster_idx"() : () -> !riscv.reg<>
        %cluster_num = "snrt.cluster_num"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.cluster_num"() : () -> !riscv.reg<>
        %is_compute_core = "snrt.is_compute_core"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.is_compute_core"() : () -> !riscv.reg<>
        %is_dm_core = "snrt.is_dm_core"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.is_dm_core"() : () -> !riscv.reg<>

        %barrier_reg_ptr = "snrt.barrier_reg_ptr"() : () -> !riscv.reg<>
        // CHECK: %{{.*}} = "snrt.barrier_reg_ptr"() : () -> !riscv.reg<>
        %global_memory0, %global_memory1 = "snrt.global_memory"() : () -> (!riscv.reg<>, !riscv.reg<>)
        // CHECK: %{{.*}}, %{{.*}} = "snrt.global_memory"() : () -> (!riscv.reg<>, !riscv.reg<>)
        %cluster_memory0, %cluster_memory1 = "snrt.cluster_memory"() : () -> (!riscv.reg<>, !riscv.reg<>)
        // CHECK: %{{.*}}, %{{.*}} = "snrt.cluster_memory"() : () -> (!riscv.reg<>, !riscv.reg<>)
        %zero_memory0, %zero_memory1 = "snrt.zero_memory"() : () -> (!riscv.reg<>, !riscv.reg<>)
        // CHECK: %{{.*}}, %{{.*}} = "snrt.zero_memory"() : () -> (!riscv.reg<>, !riscv.reg<>)

        // DMA Operations
        %dst_64 = riscv.li 100 : () -> !riscv.reg<>
        %src_64 = riscv.li 0 : () -> !riscv.reg<>
        %size = riscv.li 100 : () -> !riscv.reg<>
        %transfer_id = "snrt.dma_start_1d_wideptr"(%dst_64, %src_64, %size) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        // CHECK: %transfer_id = "snrt.dma_start_1d_wideptr"(%dst_64, %src_64, %size) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %dst_32 = riscv.li 100 : () -> !riscv.reg<>
        %src_32 = riscv.li 0 : () -> !riscv.reg<>
        %size_2 = riscv.li 100 : () -> !riscv.reg<>
        %transfer_id_2 = "snrt.dma_start_1d"(%dst_32, %src_32, %size_2) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        // CHECK: %transfer_id_2 = "snrt.dma_start_1d"(%dst_32, %src_32, %size_2) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "snrt.dma_wait"(%transfer_id) : (!riscv.reg<>) -> ()
        // CHECK: "snrt.dma_wait"(%transfer_id) : (!riscv.reg<>) -> ()
        %repeat = riscv.li 1 : () -> !riscv.reg<>
        %src_stride = riscv.li 1 : () -> !riscv.reg<>
        %dst_stride = riscv.li 1 : () -> !riscv.reg<>
        %transfer_id_3 = "snrt.dma_start_2d_wideptr"(%dst_64, %src_64, %dst_stride, %src_stride, %size, %repeat) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        // CHECK: transfer_id_3 = "snrt.dma_start_2d_wideptr"(%dst_64, %src_64, %dst_stride, %src_stride, %size, %repeat) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %transfer_id_4 = "snrt.dma_start_2d"(%dst_32, %src_32, %dst_stride, %src_stride, %size_2, %repeat) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        // CHECK: transfer_id_4 = "snrt.dma_start_2d"(%dst_32, %src_32, %dst_stride, %src_stride, %size_2, %repeat) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "snrt.dma_wait_all"() : () -> ()
        // CHECK: "snrt.dma_wait_all"() : () -> ()

        // SSR Operations
        %dm = riscv.li 0 : () -> !riscv.reg<>
        %b0 = riscv.li 100 : () -> !riscv.reg<>
        %i0 = riscv.li 101 : () -> !riscv.reg<>
        "snrt.ssr_loop_1d"(%dm, %b0, %i0) {"operandSegmentSizes" = array<i32: 1, 1, 1>} : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
        // CHECK: "snrt.ssr_loop_1d"(%dm, %b0, %i0) {"operandSegmentSizes" = array<i32: 1, 1, 1>} : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
        %b1 = riscv.li 102 : () -> !riscv.reg<>
        %i1 = riscv.li 103 : () -> !riscv.reg<>
        "snrt.ssr_loop_2d"(%dm, %b0, %b1, %i0, %i1) {"operandSegmentSizes" = array<i32: 1, 2, 2>}  : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
        // CHECK: "snrt.ssr_loop_2d"(%dm, %b0, %b1, %i0, %i1) {"operandSegmentSizes" = array<i32: 1, 2, 2>}  : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
        %b2 = riscv.li 104 : () -> !riscv.reg<>
        %i2 = riscv.li 105 : () -> !riscv.reg<>
        "snrt.ssr_loop_3d"(%dm, %b0, %b1, %b2, %i0, %i1, %i2) {"operandSegmentSizes" = array<i32: 1, 3, 3>} : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
        // CHECK: "snrt.ssr_loop_3d"(%dm, %b0, %b1, %b2, %i0, %i1, %i2) {"operandSegmentSizes" = array<i32: 1, 3, 3>} : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
        %b3 = riscv.li 106 : () -> !riscv.reg<>
        %i3 = riscv.li 107 : () -> !riscv.reg<>
        "snrt.ssr_loop_4d"(%dm, %b0, %b1, %b2, %b3, %i0, %i1, %i2, %i3) {"operandSegmentSizes" = array<i32: 1, 4, 4>} : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
        //CHECK: "snrt.ssr_loop_4d"(%dm, %b0, %b1, %b2, %b3, %i0, %i1, %i2, %i3) {"operandSegmentSizes" = array<i32: 1, 4, 4>} : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
        %dm2 = riscv.li 1 : () -> !riscv.reg<>
        %count = riscv.li 20 : () -> !riscv.reg<>
        "snrt.ssr_repeat"(%dm2, %count) : (!riscv.reg<>, !riscv.reg<>) -> ()
        // CHECK: "snrt.ssr_repeat"(%dm2, %count) : (!riscv.reg<>, !riscv.reg<>) -> ()
        "snrt.ssr_enable"() : () -> ()
        // CHECK: "snrt.ssr_enable"() : () -> ()
        "snrt.ssr_disable"() : () -> ()
        // CHECK: "snrt.ssr_disable"() : () -> ()
        %dm3 = riscv.li 2 : () -> !riscv.reg<>
        %dim = riscv.li 1 : () -> !riscv.reg<>
        %ptr = riscv.li 0 : () -> !riscv.reg<>
        "snrt.ssr_read"(%dm3, %dim, %ptr) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
        // CHECK: "snrt.ssr_read"(%dm3, %dim, %ptr) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
        "snrt.ssr_write"(%dm3, %dim, %ptr) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
        // CHECK: "snrt.ssr_write"(%dm3, %dim, %ptr) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
        "snrt.fpu_fence"() : () -> ()
        // CHECK: "snrt.fpu_fence"() : () -> ()


        "func.return"() : () -> ()
    }
}
