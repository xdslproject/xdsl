// RUN: XDSL_ROUNDTRIP

module {

func.func @graph(%arg0 : memref<32x16xi32>, %arg1 : memref<32x16xi32>) -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<32x16xi32>, memref<32x16xi32> attributes { sym_name="herd_0"} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %buf0 = memref.alloc() {sym_name = "scratch"}: memref<16x8xi32, 2>
    %buf1 = memref.alloc() {sym_name = "scratch_copy"}: memref<16x8xi32, 2>
    air.dma_memcpy_nd (%buf0[%c0, %c0][%c8, %c16][%c32, %c0], %ext0[%c8, %c0][%c8, %c16][%c32, %c0]) {id = 1 : i32} : (memref<16x8xi32, 2>, memref<32x16xi32>)
    air.dma_memcpy_nd (%ext1[%c8, %c0][%c8, %c16][%c32, %c0], %buf1[%c0, %c0][%c8, %c16][%c32, %c0]) {id = 2 : i32} : (memref<32x16xi32>, memref<16x8xi32, 2>) 
    air.herd_terminator
  }
  return
}

}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func @graph(%arg0 : memref<32x16xi32>, %arg1 : memref<32x16xi32>) {
// CHECK-NEXT:     %herd_cols = arith.constant 1 : index
// CHECK-NEXT:     %herd_rows = arith.constant 1 : index
// CHECK-NEXT:     %0 = air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, ext1 = %arg1 : memref<32x16xi32>,memref<32x16xi32>){
// CHECK-NEXT:     ^0(%ext0 : memref<32x16xi32>, %ext1 : memref<32x16xi32>):
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c128 = arith.constant 128 : index
// CHECK-NEXT:       %c32 = arith.constant 32 : index
// CHECK-NEXT:       %c16 = arith.constant 16 : index
// CHECK-NEXT:       %c8 = arith.constant 8 : index
// CHECK-NEXT:       %buf0 = memref.alloc() {"sym_name" = "scratch"} : memref<16x8xi32, 2 : i64>
// CHECK-NEXT:       %buf1 = memref.alloc() {"sym_name" = "scratch_copy"} : memref<16x8xi32, 2 : i64>
// CHECK-NEXT:       %1 = air.dma_memcpy_nd(%c0[%c0, %c0] [%c8, %c16] [%c32, %c0], %c32[%c8, %c0] [%c8, %c16] [%c32, %c0]) {"id" = 1 : i32} : (index, index)
// CHECK-NEXT:       %2 = air.dma_memcpy_nd(%c8[%c8, %c0] [%c8, %c16] [%c32, %c0], %c32[%c0, %c0] [%c8, %c16] [%c32, %c0]) {"id" = 2 : i32} : (index, index)
// CHECK-NEXT:       air.herd_terminator
// CHECK-NEXT:     }
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

#map = affine_map<()[s0] -> (s0 * 32)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)> 
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)> 
module {
  air.channel @channel_4 [2, 2]
  air.channel @channel_3 [1, 1] {broadcast_shape = [2, 1]} 
  air.channel @channel_2 [1, 1] {broadcast_shape = [2, 1]} 
  air.channel @channel_1 [1, 1] {broadcast_shape = [1, 2]} 
  air.channel @channel_0 [1, 1] {broadcast_shape = [1, 2]} 
  func.func @forward(%arg0: memref<64x128xi32>, %arg1: memref<128x64xi32>, %arg2: memref<64x64xi32>) {
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32 
    %async_token, %results = air.execute -> (memref<64x64xi32>) {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xi32>
      air.execute_terminator %alloc : memref<64x64xi32>
    }   
    %async_token_0 = air.execute [%async_token] {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xi32>
      air.execute_terminator %alloc : memref<64x64xi32>
    }   
    %async_token_1, %results_2 = air.execute -> (memref<64x64xi32>) {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xi32>
      air.execute_terminator %alloc : memref<64x64xi32>
    }   
    %async_token_3 = air.execute [%async_token_1, %async_token_0] {
        %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xi32>
        air.execute_terminator %alloc : memref<64x64xi32>
    }   
    %0 = air.wait_all async 
    %1 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %0) -> (!air.async.token) {
      %10 = air.channel.put async [%arg4]  @channel_0[] (%arg0[%c0, %arg3] [%c32, %c32] [%c128, %c1]) {id = 1 : i32} : (memref<64x128xi32>)
      scf.yield %10 : !air.async.token
    }   
    %2 = air.wait_all async 
    %3 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %2) -> (!air.async.token) {
      %10 = air.channel.put async [%arg4]  @channel_1[] (%arg0[%c32, %arg3] [%c32, %c32] [%c128, %c1]) {id = 2 : i32} : (memref<64x128xi32>)
      scf.yield %10 : !air.async.token
    }   
    %4 = air.wait_all async 
    %5 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %4) -> (!air.async.token) {
      %10 = air.channel.put async [%arg4]  @channel_2[] (%arg1[%arg3, %c0] [%c32, %c32] [%c64, %c1]) {id = 3 : i32} : (memref<128x64xi32>)
      scf.yield %10 : !air.async.token
    }   
    %6 = air.wait_all async 
    %7 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %6) -> (!air.async.token) {
      %10 = air.channel.put async [%arg4]  @channel_3[] (%arg1[%arg3, %c32] [%c32, %c32] [%c64, %c1]) {id = 4 : i32} : (memref<128x64xi32>)
      scf.yield %10 : !air.async.token
    }   
    %10 = air.channel.get async [%async_token_3]  @channel_4[] (%results_2[] [%c32, %c32] [%c64, %c1]) {id = 5 : i32} : (memref<64x64xi32>)
    return
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   air.channel @channel_4 [2 : i64, 2 : i64]
// CHECK-NEXT:   air.channel @channel_3 [1 : i64, 1 : i64] {"broadcast_shape" = [2 : i64, 1 : i64]}
// CHECK-NEXT:   air.channel @channel_2 [1 : i64, 1 : i64] {"broadcast_shape" = [2 : i64, 1 : i64]}
// CHECK-NEXT:   air.channel @channel_1 [1 : i64, 1 : i64] {"broadcast_shape" = [1 : i64, 2 : i64]}
// CHECK-NEXT:   air.channel @channel_0 [1 : i64, 1 : i64] {"broadcast_shape" = [1 : i64, 2 : i64]}
// CHECK-NEXT:   func.func @forward(%arg0 : memref<64x128xi32>, %arg1 : memref<128x64xi32>, %arg2 : memref<64x64xi32>) {
// CHECK-NEXT:     %c64 = arith.constant 64 : index
// CHECK-NEXT:     %c32 = arith.constant 32 : index
// CHECK-NEXT:     %c128 = arith.constant 128 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c2 = arith.constant 2 : index
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %async_token, %results = "air.execute"() ({
// CHECK-NEXT:       %alloc = memref.alloc() {"alignment" = 64 : i64} : memref<64x64xi32>
// CHECK-NEXT:       "air.execute_terminator"(%alloc) : (memref<64x64xi32>) -> ()
// CHECK-NEXT:     }) : () -> (!air.async.token, memref<64x64xi32>)
// CHECK-NEXT:     %async_token_0 = "air.execute"(%async_token) ({
// CHECK-NEXT:       %alloc_1 = memref.alloc() {"alignment" = 64 : i64} : memref<64x64xi32>
// CHECK-NEXT:       "air.execute_terminator"(%alloc_1) : (memref<64x64xi32>) -> ()
// CHECK-NEXT:     }) : (!air.async.token) -> !air.async.token
// CHECK-NEXT:     %async_token_1, %results_2 = "air.execute"() ({
// CHECK-NEXT:       %alloc_2 = memref.alloc() {"alignment" = 64 : i64} : memref<64x64xi32>
// CHECK-NEXT:       "air.execute_terminator"(%alloc_2) : (memref<64x64xi32>) -> ()
// CHECK-NEXT:     }) : () -> (!air.async.token, memref<64x64xi32>)
// CHECK-NEXT:     %async_token_3 = "air.execute"(%async_token_1, %async_token_0) ({
// CHECK-NEXT:       %alloc_3 = memref.alloc() {"alignment" = 64 : i64} : memref<64x64xi32>
// CHECK-NEXT:       "air.execute_terminator"(%alloc_3) : (memref<64x64xi32>) -> ()
// CHECK-NEXT:     }) : (!air.async.token, !air.async.token) -> !air.async.token
// CHECK-NEXT:     %0 = "air.wait_all"() : () -> !air.async.token
// CHECK-NEXT:     %1 = scf.for %arg3 = %c0 to %c128 step %c32 iter_args(%arg4 = %0) -> (!air.async.token) {
// CHECK-NEXT:       %2 = air.channel.put async[%arg4] @channel_0[] (%c0[%c0, %arg3] [%c32, %c32] [%c128, %c1]) {"id" = 1 : i32} : (index)
// CHECK-NEXT:       scf.yield %2 : !air.async.token
// CHECK-NEXT:     }
// CHECK-NEXT:     %3 = "air.wait_all"() : () -> !air.async.token
// CHECK-NEXT:     %4 = scf.for %arg3_1 = %c0 to %c128 step %c32 iter_args(%arg4_1 = %3) -> (!air.async.token) {
// CHECK-NEXT:       %5 = air.channel.put async[%arg4_1] @channel_1[] (%c32[%c32, %arg3_1] [%c32, %c32] [%c128, %c1]) {"id" = 2 : i32} : (index)
// CHECK-NEXT:       scf.yield %5 : !air.async.token
// CHECK-NEXT:     }
// CHECK-NEXT:     %6 = "air.wait_all"() : () -> !air.async.token
// CHECK-NEXT:     %7 = scf.for %arg3_2 = %c0 to %c128 step %c32 iter_args(%arg4_2 = %6) -> (!air.async.token) {
// CHECK-NEXT:       %8 = air.channel.put async[%arg4_2] @channel_2[] (%arg3_2[%arg3_2, %c0] [%c32, %c32] [%c64, %c1]) {"id" = 3 : i32} : (index)
// CHECK-NEXT:       scf.yield %8 : !air.async.token
// CHECK-NEXT:     }
// CHECK-NEXT:     %9 = "air.wait_all"() : () -> !air.async.token
// CHECK-NEXT:     %10 = scf.for %arg3_3 = %c0 to %c128 step %c32 iter_args(%arg4_3 = %9) -> (!air.async.token) {
// CHECK-NEXT:       %11 = air.channel.put async[%arg4_3] @channel_3[] (%arg3_3[%arg3_3, %c32] [%c32, %c32] [%c64, %c1]) {"id" = 4 : i32} : (index)
// CHECK-NEXT:       scf.yield %11 : !air.async.token
// CHECK-NEXT:     }
// CHECK-NEXT:     %12 = air.channel.get async[%async_token_3] @channel_4[] (%c32[] [%c32, %c32] [%c64, %c1]) {"id" = 5 : i32} : (index)
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }