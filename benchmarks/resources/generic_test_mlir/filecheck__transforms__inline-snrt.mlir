"builtin.module"() ({
  %0 = "snrt.global_core_base_hartid"() : () -> i32
  %1 = "snrt.global_core_idx"() : () -> i32
  %2 = "snrt.global_core_num"() : () -> i32
  %3 = "snrt.cluster_core_idx"() : () -> i32
  %4 = "snrt.cluster_core_num"() : () -> i32
  %5 = "snrt.cluster_compute_core_num"() : () -> i32
  %6 = "snrt.cluster_dm_core_num"() : () -> i32
  %7 = "snrt.cluster_idx"() : () -> i32
  %8 = "snrt.cluster_num"() : () -> i32
  %9 = "snrt.is_compute_core"() : () -> i1
  %10 = "snrt.is_dm_core"() : () -> i1
  "snrt.cluster_hw_barrier"() : () -> ()
  "snrt.ssr_disable"() : () -> ()
  %11:3 = "test.op"() : () -> (i32, i32, i32)
  %12 = "snrt.dma_start_1d"(%11#0, %11#1, %11#2) : (i32, i32, i32) -> i32
  %13:2 = "test.op"() : () -> (i64, i64)
  %14 = "snrt.dma_start_1d_wideptr"(%13#0, %13#1, %11#2) : (i64, i64, i32) -> i32
  %15:3 = "test.op"() : () -> (i32, i32, i32)
  %16 = "snrt.dma_start_2d_wideptr"(%13#0, %13#1, %15#0, %15#1, %11#2, %15#2) : (i64, i64, i32, i32, i32, i32) -> i32
  %17 = "snrt.dma_start_2d"(%11#0, %11#1, %15#0, %15#1, %11#2, %15#2) : (i32, i32, i32, i32, i32, i32) -> i32
}) : () -> ()
