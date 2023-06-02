// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | filecheck %s
"builtin.module"() ({
    // Printf

    // Pointer to format string
    "llvm.mlir.global"() ({}) {"global_type" = !llvm.array<6 x i8>, "sym_name" = "format_str", "linkage" = #llvm.linkage<"internal">, "addr_space" = 0 : i32, "constant", "value" = "%d:%d!", "unnamed_addr" = 0 : i64} : () -> ()
    %format_str_ptr = "llvm.mlir.addressof"() {"global_name" = @format_str} : () -> !llvm.ptr<i8>
    // Values to print
    %element1 = "arith.constant"() {"value" = 2 : si32} : () -> si32
    %element2 = "arith.constant"() {"value" = 0 : si32} : () -> si32
    %count = "snrt.printf"(%format_str_ptr, %element1, %element2) : (!llvm.ptr<i8>, si32, si32) -> i32
    // CHECK: %count = "snrt.printf"(%format_str_ptr, %element1, %element2) : (!llvm.ptr<i8>, si32, si32) -> i32


    // Barriers
    "snrt.cluster_hw_barrier"() : () -> ()
    // CHECK: "snrt.cluster_hw_barrier"() : () -> ()

    // Runtime Info Getters
    %cluster_num = "snrt.cluster_num"() : () -> ui32
    // CHECK: %{{.*}} = "snrt.cluster_num"() : () -> ui32

    // DMA Operations
    %dst_64 = "arith.constant"() {"value" = 100 : ui64} : () -> ui64
    %src_64 = "arith.constant"() {"value" = 0 : ui64} : () -> ui64
    %size = "arith.constant"() {"value" = 100 : index} : () -> index
    %transfer_id = "snrt.dma_start_1d_wideptr"(%dst_64, %src_64, %size) : (ui64, ui64, index) -> ui32
    // CHECK: %transfer_id = "snrt.dma_start_1d_wideptr"(%dst_64, %src_64, %size) : (ui64, ui64, index) -> ui32
    %dst_32 = "arith.constant"() {"value" = 100: ui32} : () -> ui32
    %src_32 = "arith.constant"() {"value" = 0: ui32} : () -> ui32
    %size_2 = "arith.constant"() {"value" = 100: index} : () -> index
    %transfer_id_2 = "snrt.dma_start_1d"(%dst_32, %src_32, %size_2) : (ui32, ui32, index) -> ui32
    // CHECK: %transfer_id_2 = "snrt.dma_start_1d"(%dst_32, %src_32, %size_2) : (ui32, ui32, index) -> ui32
    "snrt.dma_wait"(%transfer_id) : (ui32) -> ()
    // CHECK: "snrt.dma_wait"(%transfer_id) : (ui32) -> ()
    %repeat = "arith.constant"() {"value" = 1: index} : () -> index
    %src_stride = "arith.constant"() {"value" = 1: index} : () -> index
    %dst_stride = "arith.constant"() {"value" = 1: index} : () -> index
    %transfer_id_3 = "snrt.dma_start_2d_wideptr"(%dst_64, %src_64, %dst_stride, %src_stride, %size, %repeat) : (ui64, ui64, index, index, index, index) -> ui32
    // CHECK: transfer_id_3 = "snrt.dma_start_2d_wideptr"(%dst_64, %src_64, %dst_stride, %src_stride, %size, %repeat) : (ui64, ui64, index, index, index, index) -> ui32
    %transfer_id_4 = "snrt.dma_start_2d"(%dst_32, %src_32, %dst_stride, %src_stride, %size_2, %repeat) : (ui32, ui32, index, index, index, index) -> ui32
    // CHECK: transfer_id_4 = "snrt.dma_start_2d"(%dst_32, %src_32, %dst_stride, %src_stride, %size_2, %repeat) : (ui32, ui32, index, index, index, index) -> ui32
    "snrt.dma_wait_all"() : () -> ()
    // CHECK: "snrt.dma_wait_all"() : () -> ()

}) : () -> ()
