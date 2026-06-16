// RUN: XDSL_ROUNDTRIP

shard.grid @grid0(shape = 2x2x4)
shard.grid @grid1(shape = 4x?)
shard.grid @grid2(shape = ?x4)
shard.grid @grid3(shape = ?x?)
shard.grid @grid4(shape = 3)
shard.grid @grid5(shape = ?)

// CHECK:      shard.grid @grid0(shape = 2x2x4)
// CHECK-NEXT: shard.grid @grid1(shape = 4x?)
// CHECK-NEXT: shard.grid @grid2(shape = ?x4)
// CHECK-NEXT: shard.grid @grid3(shape = ?x?)
// CHECK-NEXT: shard.grid @grid4(shape = 3)
// CHECK-NEXT: shard.grid @grid5(shape = ?)

%0 = shard.sharding @grid0 split_axes = [[]] : !shard.sharding
// CHECK-NEXT: %0 = shard.sharding @grid0 split_axes = [[]] : !shard.sharding
%1 = shard.sharding @grid0 split_axes = [[0]] : !shard.sharding
// CHECK-NEXT: %1 = shard.sharding @grid0 split_axes = [[0]] : !shard.sharding
%2 = shard.sharding @grid0 split_axes = [[1]] : !shard.sharding
// CHECK-NEXT: %2 = shard.sharding @grid0 split_axes = [[1]] : !shard.sharding
%3 = shard.sharding @grid1 split_axes = [[], [0]] : !shard.sharding
// CHECK-NEXT: %3 = shard.sharding @grid1 split_axes = [[], [0]] : !shard.sharding
%4 = shard.sharding @grid3 split_axes = [[0], [], [1]] : !shard.sharding
// CHECK-NEXT: %4 = shard.sharding @grid3 split_axes = [[0], [], [1]] : !shard.sharding
%5 = shard.sharding @grid4 split_axes = [[0]] halo_sizes = [1, 4] : !shard.sharding
// CHECK-NEXT: %5 = shard.sharding @grid4 split_axes = [[0]] halo_sizes = [1, 4] : !shard.sharding

%6 = arith.constant 3 : i64
// CHECK-NEXT: %6 = arith.constant 3 : i64
%7 = shard.sharding @grid4 split_axes = [[0]] halo_sizes = [4, %6] : !shard.sharding
// CHECK-NEXT: %7 = shard.sharding @grid4 split_axes = [[0]] halo_sizes = [4, %6] : !shard.sharding
%8 = shard.sharding @grid4 split_axes = [[0]] sharded_dims_offsets = [0, 1, 4, 6] : !shard.sharding
// CHECK-NEXT: %8 = shard.sharding @grid4 split_axes = [[0]] sharded_dims_offsets = [0, 1, 4, 6] : !shard.sharding
%9 = shard.sharding @grid4 split_axes = [[0]] sharded_dims_offsets = [0, 2, %6, 5] : !shard.sharding
// CHECK-NEXT: %9 = shard.sharding @grid4 split_axes = [[0]] sharded_dims_offsets = [0, 2, %6, 5] : !shard.sharding

// Collective communications
%t = "test.op"() : () -> tensor<2x2xi8>
// CHECK-NEXT: %t = "test.op"() : () -> tensor<2x2xi8>
%10 = shard.broadcast %t on @grid0 grid_axes = [0] root = [0] : (tensor<2x2xi8>) -> tensor<2x2xi8>
// CHECK-NEXT: %10 = shard.broadcast %t on @grid0 grid_axes = [0] root = [0] : (tensor<2x2xi8>) -> tensor<2x2xi8>
%11 = shard.gather %t on @grid0 grid_axes = [1] gather_axis = 1 root = [1] : (tensor<2x2xi8>) -> tensor<2x4xi8>
// CHECK-NEXT: %11 = shard.gather %t on @grid0 grid_axes = [1] gather_axis = 1 root = [1] : (tensor<2x2xi8>) -> tensor<2x4xi8>
%12 = shard.scatter %t on @grid0 grid_axes = [0] scatter_axis = 0 root = [1] : (tensor<2x2xi8>) -> tensor<1x2xi8>
// CHECK-NEXT: %12 = shard.scatter %t on @grid0 grid_axes = [0] scatter_axis = 0 root = [1] : (tensor<2x2xi8>) -> tensor<1x2xi8>
%13 = shard.recv %t on @grid0 grid_axes = [0, 2] source = [0, 1] : (tensor<2x2xi8>) -> tensor<2x2xi8>
// CHECK-NEXT: %13 = shard.recv %t on @grid0 grid_axes = [0, 2] source = [0, 1] : (tensor<2x2xi8>) -> tensor<2x2xi8>
%14 = shard.send %t on @grid0 grid_axes = [1] destination = [1] : (tensor<2x2xi8>) -> tensor<2x2xi8>
// CHECK-NEXT: %14 = shard.send %t on @grid0 grid_axes = [1] destination = [1] : (tensor<2x2xi8>) -> tensor<2x2xi8>
%15 = shard.shift %t on @grid0 grid_axes = [1] shift_axis = 1 offset = 2 rotate : tensor<2x2xi8> -> tensor<2x2xi8>
// CHECK-NEXT: %15 = shard.shift %t on @grid0 grid_axes = [1] shift_axis = 1 offset = 2 rotate : tensor<2x2xi8> -> tensor<2x2xi8>

// Sharding
%16 = shard.shard %t to %7 : tensor<2x2xi8>
// CHECK-NEXT: %16 = shard.shard %t to %7 : tensor<2x2xi8>
%17 = shard.shard %t to %7 annotate_for_users : tensor<2x2xi8>
// CHECK-NEXT: %17 = shard.shard %t to %7 annotate_for_users : tensor<2x2xi8>
