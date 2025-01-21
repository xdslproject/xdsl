"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    "snrt.cluster_hw_barrier"() : () -> ()
    "snrt.cluster_sw_barrier"() : () -> ()
    "snrt.global_barrier"() : () -> ()
    %0 = "snrt.global_core_base_hartid"() : () -> i32
    %1 = "snrt.global_core_idx"() : () -> i32
    %2 = "snrt.global_core_num"() : () -> i32
    %3 = "snrt.global_compute_core_idx"() : () -> i32
    %4 = "snrt.global_compute_core_num"() : () -> i32
    %5 = "snrt.global_dm_core_num"() : () -> i32
    %6 = "snrt.cluster_core_idx"() : () -> i32
    %7 = "snrt.cluster_core_num"() : () -> i32
    %8 = "snrt.cluster_compute_core_idx"() : () -> i32
    %9 = "snrt.cluster_compute_core_num"() : () -> i32
    %10 = "snrt.cluster_dm_core_idx"() : () -> i32
    %11 = "snrt.cluster_dm_core_num"() : () -> i32
    %12 = "snrt.cluster_idx"() : () -> i32
    %13 = "snrt.cluster_num"() : () -> i32
    %14 = "snrt.is_compute_core"() : () -> i1
    %15 = "snrt.is_dm_core"() : () -> i1
    %16 = "snrt.barrier_reg_ptr"() : () -> i32
    %17:2 = "snrt.global_memory"() : () -> (i64, i64)
    %18:2 = "snrt.cluster_memory"() : () -> (i64, i64)
    %19:2 = "snrt.zero_memory"() : () -> (i64, i64)
    %20 = "arith.constant"() <{value = 100 : i64}> : () -> i64
    %21 = "arith.constant"() <{value = 0 : i64}> : () -> i64
    %22 = "arith.constant"() <{value = 100 : i32}> : () -> i32
    %23 = "snrt.dma_start_1d_wideptr"(%20, %21, %22) : (i64, i64, i32) -> i32
    %24 = "arith.constant"() <{value = 100 : i32}> : () -> i32
    %25 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %26 = "arith.constant"() <{value = 100 : i32}> : () -> i32
    %27 = "snrt.dma_start_1d"(%24, %25, %22) : (i32, i32, i32) -> i32
    "snrt.dma_wait"(%23) : (i32) -> ()
    %28 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %29 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %30 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %31 = "snrt.dma_start_2d_wideptr"(%20, %21, %30, %29, %26, %28) : (i64, i64, i32, i32, i32, i32) -> i32
    %32 = "snrt.dma_start_2d"(%24, %25, %30, %29, %26, %28) : (i32, i32, i32, i32, i32, i32) -> i32
    "snrt.dma_wait_all"() : () -> ()
    %33 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %34 = "arith.constant"() <{value = 100 : index}> : () -> index
    %35 = "arith.constant"() <{value = 101 : index}> : () -> index
    "snrt.ssr_loop_1d"(%33, %34, %35) {operandSegmentSizes = array<i32: 1, 1, 1>} : (i32, index, index) -> ()
    %36 = "arith.constant"() <{value = 102 : index}> : () -> index
    %37 = "arith.constant"() <{value = 103 : index}> : () -> index
    "snrt.ssr_loop_2d"(%33, %34, %36, %35, %37) {operandSegmentSizes = array<i32: 1, 2, 2>} : (i32, index, index, index, index) -> ()
    %38 = "arith.constant"() <{value = 104 : index}> : () -> index
    %39 = "arith.constant"() <{value = 105 : index}> : () -> index
    "snrt.ssr_loop_3d"(%33, %34, %36, %38, %35, %37, %39) {operandSegmentSizes = array<i32: 1, 3, 3>} : (i32, index, index, index, index, index, index) -> ()
    %40 = "arith.constant"() <{value = 106 : index}> : () -> index
    %41 = "arith.constant"() <{value = 107 : index}> : () -> index
    "snrt.ssr_loop_4d"(%33, %34, %36, %38, %40, %35, %37, %39, %41) {operandSegmentSizes = array<i32: 1, 4, 4>} : (i32, index, index, index, index, index, index, index, index) -> ()
    %42 = "arith.constant"() <{value = 20 : i32}> : () -> i32
    "snrt.ssr_repeat"(%42) <{dm = 3 : i32}> : (i32) -> ()
    "snrt.ssr_enable"() : () -> ()
    "snrt.ssr_disable"() : () -> ()
    %43 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %44 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %45 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "snrt.ssr_read"(%45) <{dim = 1 : i32, dm = 0 : i32}> : (i32) -> ()
    "snrt.ssr_write"(%45) <{dim = 1 : i32, dm = 0 : i32}> : (i32) -> ()
    "snrt.fpu_fence"() : () -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
