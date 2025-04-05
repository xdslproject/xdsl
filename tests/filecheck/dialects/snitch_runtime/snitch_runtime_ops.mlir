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
        %global_core_base_hartid = "snrt.global_core_base_hartid"() : () -> i32
        // CHECK: %{{.*}} = "snrt.global_core_base_hartid"() : () -> i32
        %global_core_idx = "snrt.global_core_idx"() : () -> i32
        // CHECK: %{{.*}} = "snrt.global_core_idx"() : () -> i32
        %global_core_num = "snrt.global_core_num"() : () -> i32
        // CHECK: %{{.*}} = "snrt.global_core_num"() : () -> i32
        %global_compute_core_idx = "snrt.global_compute_core_idx"() : () -> i32
        // CHECK: %{{.*}} = "snrt.global_compute_core_idx"() : () -> i32
        %global_compute_core_num = "snrt.global_compute_core_num"() : () -> i32
        // CHECK: %{{.*}} = "snrt.global_compute_core_num"() : () -> i32
        %global_dm_core_num = "snrt.global_dm_core_num"() : () -> i32
        // CHECK: %{{.*}} = "snrt.global_dm_core_num"() : () -> i32
        %gcluster_core_idx = "snrt.cluster_core_idx"() : () -> i32
        // CHECK: %{{.*}} = "snrt.cluster_core_idx"() : () -> i32
        %cluster_core_num = "snrt.cluster_core_num"() : () -> i32
        // CHECK: %{{.*}} = "snrt.cluster_core_num"() : () -> i32
        %cluster_compute_core_idx = "snrt.cluster_compute_core_idx"() : () -> i32
        // CHECK: %{{.*}} = "snrt.cluster_compute_core_idx"() : () -> i32
        %cluster_compute_core_num = "snrt.cluster_compute_core_num"() : () -> i32
        // CHECK: %{{.*}} = "snrt.cluster_compute_core_num"() : () -> i32
        %cluster_dm_core_idx = "snrt.cluster_dm_core_idx"() : () -> i32
        // CHECK: %{{.*}} = "snrt.cluster_dm_core_idx"() : () -> i32
        %cluster_dm_core_num = "snrt.cluster_dm_core_num"() : () -> i32
        // CHECK: %{{.*}} = "snrt.cluster_dm_core_num"() : () -> i32
        %cluster_idx = "snrt.cluster_idx"() : () -> i32
        // CHECK: %{{.*}} = "snrt.cluster_idx"() : () -> i32
        %cluster_num = "snrt.cluster_num"() : () -> i32
        // CHECK: %{{.*}} = "snrt.cluster_num"() : () -> i32
        %is_compute_core = "snrt.is_compute_core"() : () -> i1
        // CHECK: %{{.*}} = "snrt.is_compute_core"() : () -> i1
        %is_dm_core = "snrt.is_dm_core"() : () -> i1
        // CHECK: %{{.*}} = "snrt.is_dm_core"() : () -> i1

        %barrier_reg_ptr = "snrt.barrier_reg_ptr"() : () -> i32
        // CHECK: %{{.*}} = "snrt.barrier_reg_ptr"() : () -> i32
        %global_memory0, %global_memory1 = "snrt.global_memory"() : () -> (i64, i64)
        // CHECK: %{{.*}}, %{{.*}} = "snrt.global_memory"() : () -> (i64, i64)
        %cluster_memory0, %cluster_memory1 = "snrt.cluster_memory"() : () -> (i64, i64)
        // CHECK: %{{.*}}, %{{.*}} = "snrt.cluster_memory"() : () -> (i64, i64)
        %zero_memory0, %zero_memory1 = "snrt.zero_memory"() : () -> (i64, i64)
        // CHECK: %{{.*}}, %{{.*}} = "snrt.zero_memory"() : () -> (i64, i64)

        // DMA Operations
        %dst_i64 = arith.constant 100 : i64
        %src_i64 = arith.constant 0 : i64
        %size = arith.constant 100 : i32
        %transfer_id = "snrt.dma_start_1d_wideptr"(%dst_i64, %src_i64, %size) : (i64, i64, i32) -> i32
        // CHECK: %transfer_id = "snrt.dma_start_1d_wideptr"(%dst_i64, %src_i64, %size) : (i64, i64, i32) -> i32
        %dst_i32 = arith.constant 100 : i32
        %src_i32 = arith.constant 0 : i32
        %size_1 = arith.constant 100 : i32
        %transfer_id_1 = "snrt.dma_start_1d"(%dst_i32, %src_i32, %size) : (i32, i32, i32) -> i32
        // CHECK: %transfer_id_1 = "snrt.dma_start_1d"(%dst_i32, %src_i32, %size) : (i32, i32, i32) -> i32
        "snrt.dma_wait"(%transfer_id) : (i32) -> ()
        // CHECK: "snrt.dma_wait"(%transfer_id) : (i32) -> ()
        %repeat = arith.constant 1 : i32
        %src_stride = arith.constant 1 : i32
        %dst_stride = arith.constant 1 : i32
        %transfer_id_2 = "snrt.dma_start_2d_wideptr"(%dst_i64, %src_i64, %dst_stride, %src_stride, %size_1, %repeat) : (i64, i64, i32, i32, i32, i32) -> i32
        // CHECK: transfer_id_2 = "snrt.dma_start_2d_wideptr"(%dst_i64, %src_i64, %dst_stride, %src_stride, %size_1, %repeat) : (i64, i64, i32, i32, i32, i32) -> i32
        %transfer_id_3 = "snrt.dma_start_2d"(%dst_i32, %src_i32, %dst_stride, %src_stride, %size_1, %repeat) : (i32, i32, i32, i32, i32, i32) -> i32
        // CHECK: transfer_id_3 = "snrt.dma_start_2d"(%dst_i32, %src_i32, %dst_stride, %src_stride, %size_1, %repeat) : (i32, i32, i32, i32, i32, i32) -> i32
        "snrt.dma_wait_all"() : () -> ()
        // CHECK: "snrt.dma_wait_all"() : () -> ()

        // SSR Operations
        %dm = arith.constant 0 : i32
        %b0 = arith.constant 100 : index
        %i0 = arith.constant 101 : index
        "snrt.ssr_loop_1d"(%dm, %b0, %i0) {operandSegmentSizes = array<i32: 1, 1, 1>} : (i32, index, index) -> ()
        // CHECK: "snrt.ssr_loop_1d"(%dm, %b0, %i0) {operandSegmentSizes = array<i32: 1, 1, 1>} : (i32, index, index) -> ()
        %b1 = arith.constant 102 : index
        %i1 = arith.constant 103 : index
        "snrt.ssr_loop_2d"(%dm, %b0, %b1, %i0, %i1) {operandSegmentSizes = array<i32: 1, 2, 2>}  : (i32, index, index, index, index) -> ()
        // CHECK: "snrt.ssr_loop_2d"(%dm, %b0, %b1, %i0, %i1) {operandSegmentSizes = array<i32: 1, 2, 2>}  : (i32, index, index, index, index) -> ()
        %b2 = arith.constant 104 : index
        %i2 = arith.constant 105 : index
        "snrt.ssr_loop_3d"(%dm, %b0, %b1, %b2, %i0, %i1, %i2) {operandSegmentSizes = array<i32: 1, 3, 3>} : (i32, index, index, index, index, index, index) -> ()
        // CHECK: "snrt.ssr_loop_3d"(%dm, %b0, %b1, %b2, %i0, %i1, %i2) {operandSegmentSizes = array<i32: 1, 3, 3>} : (i32, index, index, index, index, index, index) -> ()
        %b3 = arith.constant 106 : index
        %i3 = arith.constant 107 : index
        "snrt.ssr_loop_4d"(%dm, %b0, %b1, %b2, %b3, %i0, %i1, %i2, %i3) {operandSegmentSizes = array<i32: 1, 4, 4>} : (i32, index, index, index, index, index, index, index, index) -> ()
        //CHECK: "snrt.ssr_loop_4d"(%dm, %b0, %b1, %b2, %b3, %i0, %i1, %i2, %i3) {operandSegmentSizes = array<i32: 1, 4, 4>} : (i32, index, index, index, index, index, index, index, index) -> ()
        %count = arith.constant 20 : i32
        "snrt.ssr_repeat"(%count) <{dm = 3 : i32}> : (i32) -> ()
        // CHECK: "snrt.ssr_repeat"(%count) <{dm = 3 : i32}> : (i32) -> ()
        "snrt.ssr_enable"() : () -> ()
        // CHECK: "snrt.ssr_enable"() : () -> ()
        "snrt.ssr_disable"() : () -> ()
        // CHECK: "snrt.ssr_disable"() : () -> ()
        %dm3 = arith.constant 2 : i32
        %dim = arith.constant 1 : i32
        %ptr = arith.constant 0 : i32
        "snrt.ssr_read"(%ptr) <{dm = 0 : i32, dim = 1 : i32}> : (i32) -> ()
        // CHECK: "snrt.ssr_read"(%ptr) <{dm = 0 : i32, dim = 1 : i32}> : (i32) -> ()
        "snrt.ssr_write"(%ptr) <{dm = 0 : i32, dim = 1 : i32}> : (i32) -> ()
        // CHECK: "snrt.ssr_write"(%ptr) <{dm = 0 : i32, dim = 1 : i32}> : (i32) -> ()
        "snrt.fpu_fence"() : () -> ()
        // CHECK: "snrt.fpu_fence"() : () -> ()


        "func.return"() : () -> ()
    }
}
