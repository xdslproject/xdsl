// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

%dst_i64 = arith.constant 100 : i64
%src_i64 = arith.constant 0 : i64
%size = arith.constant 100 : i32
%transfer_id = "snrt.dma_start_1d"(%dst_i64, %src_i64, %size) : (i64, i64, i32) -> i32
// CHECK: Invalid value 64, expected 32

// -----
%dst_i32 = arith.constant 100 : i32
%src_i32 = arith.constant 0 : i32
%size = arith.constant 100 : i32
%transfer_id_1 = "snrt.dma_start_1d_wideptr"(%dst_i32, %src_i32, %size) : (i32, i32, i32) -> i32
// CHECK: Invalid value 32, expected 64
