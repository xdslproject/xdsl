// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

shard.grid @mesh0(shape = 2x2x4)
// CHECK:      shard.grid @mesh0(shape = 2x2x4)
shard.grid @mesh1(shape = 4x?)
// CHECK-NEXT: shard.grid @mesh1(shape = 4x?)
shard.grid @mesh2(shape = ?x4)
// CHECK-NEXT: shard.grid @mesh2(shape = ?x4)
shard.grid @mesh3(shape = ?x?)
// CHECK-NEXT: shard.grid @mesh3(shape = ?x?)
shard.grid @mesh4(shape = 3)
// CHECK-NEXT: shard.grid @mesh4(shape = 3)
shard.grid @mesh5(shape = ?)
// CHECK-NEXT: shard.grid @mesh5(shape = ?)

%0 = shard.sharding @mesh0 split_axes = [[]] : !shard.sharding
// CHECK-NEXT: {{%.*}} = shard.sharding @mesh0 split_axes = [[]] : !shard.sharding
%1 = shard.sharding @mesh0 split_axes = [[0]] : !shard.sharding
// CHECK-NEXT: {{%.*}} = shard.sharding @mesh0 split_axes = [[0]] : !shard.sharding
%2 = shard.sharding @mesh0 split_axes = [[1]] : !shard.sharding
// CHECK-NEXT: {{%.*}} = shard.sharding @mesh0 split_axes = [[1]] : !shard.sharding
%3 = shard.sharding @mesh1 split_axes = [[], [0]] : !shard.sharding
// CHECK-NEXT: {{%.*}} = shard.sharding @mesh1 split_axes = [[], [0]] : !shard.sharding
%4 = shard.sharding @mesh3 split_axes = [[0], [], [1]] : !shard.sharding
// CHECK-NEXT: {{%.*}} = shard.sharding @mesh3 split_axes = [[0], [], [1]] : !shard.sharding
%9 = shard.sharding @mesh4 split_axes = [[0]] halo_sizes = [1, 4] : !shard.sharding
// CHECK-NEXT: {{%.*}} = shard.sharding @mesh4 split_axes = [[0]] halo_sizes = [1, 4] : !shard.sharding

%10 = arith.constant 3 : i64
// CHECK-NEXT: {{%.*}} = arith.constant 3 : i64
%11 = shard.sharding @mesh4 split_axes = [[0]] halo_sizes = [4, %10] : !shard.sharding
// CHECK-NEXT: {{%.*}} = shard.sharding @mesh4 split_axes = [[0]] halo_sizes = [4, {{%.*}}] : !shard.sharding
%12 = shard.sharding @mesh4 split_axes = [[0]] sharded_dims_offsets = [0, 1, 4, 6] : !shard.sharding
// CHECK-NEXT: {{%.*}} = shard.sharding @mesh4 split_axes = [[0]] sharded_dims_offsets = [0, 1, 4, 6] : !shard.sharding
%13 = shard.sharding @mesh4 split_axes = [[0]] sharded_dims_offsets = [0, 2, %10, 5] : !shard.sharding
// CHECK-NEXT: {{%.*}} = shard.sharding @mesh4 split_axes = [[0]] sharded_dims_offsets = [0, 2, {{%.*}}, 5] : !shard.sharding

// Collective communication
%t = "test.op"() : () -> tensor<2x2xi8>
// CHECK-NEXT: {{%.*}} = "test.op"() : () -> tensor<2x2xi8>
%14 = shard.broadcast %t on @mesh0 grid_axes = [0] root = [0] : (tensor<2x2xi8>) -> tensor<2x2xi8>
// CHECK-NEXT: {{%.*}} = shard.broadcast {{%.*}} on @mesh0 grid_axes = [0] root = [0] : (tensor<2x2xi8>) -> tensor<2x2xi8>
%15 = shard.gather %t on @mesh0 grid_axes = [1] gather_axis = 1 root = [1] : (tensor<2x2xi8>) -> tensor<2x4xi8>
// CHECK-NEXT: {{%.*}} = shard.gather {{%.*}} on @mesh0 grid_axes = [1] gather_axis = 1 root = [1] : (tensor<2x2xi8>) -> tensor<2x4xi8>
%16 = shard.scatter %t on @mesh0 grid_axes = [0] scatter_axis = 0 root = [1] : (tensor<2x2xi8>) -> tensor<1x2xi8>
// CHECK-NEXT: {{%.*}} = shard.scatter {{%.*}} on @mesh0 grid_axes = [0] scatter_axis = 0 root = [1] : (tensor<2x2xi8>) -> tensor<1x2xi8>
%17 = shard.recv %t on @mesh0 grid_axes = [0, 2] source = [0, 1] : (tensor<2x2xi8>) -> tensor<2x2xi8>
// CHECK-NEXT: {{%.*}} = shard.recv {{%.*}} on @mesh0 grid_axes = [0, 2] source = [0, 1] : (tensor<2x2xi8>) -> tensor<2x2xi8>
%18 = shard.send %t on @mesh0 grid_axes = [1] destination = [1] : (tensor<2x2xi8>) -> tensor<2x2xi8>
// CHECK-NEXT: {{%.*}} = shard.send {{%.*}} on @mesh0 grid_axes = [1] destination = [1] : (tensor<2x2xi8>) -> tensor<2x2xi8>
%19 = shard.shift %t on @mesh0 grid_axes = [1] shift_axis = 1 offset = 2 rotate : tensor<2x2xi8> -> tensor<2x2xi8>
// CHECK-NEXT: {{%.*}} = shard.shift {{%.*}} on @mesh0 grid_axes = [1] shift_axis = 1 offset = 2 rotate : tensor<2x2xi8> -> tensor<2x2xi8>

// Sharding
%20 = shard.shard %t to %11 : tensor<2x2xi8>
// CHECK-NEXT: %{{.*}} = shard.shard %{{.*}} to %{{.*}} : tensor<2x2xi8>
%21 = shard.shard %t to %11 annotate_for_users : tensor<2x2xi8>
// CHECK-NEXT: %{{.*}} = shard.shard %{{.*}} to %{{.*}} annotate_for_users : tensor<2x2xi8>
