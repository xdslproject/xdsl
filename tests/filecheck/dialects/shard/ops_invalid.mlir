// RUN: xdsl-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect | filecheck %s

shard.grid @grid0(shape = [])

// CHECK: 'shard.grid' op rank of grid is expected to be a positive integer 

// -----


shard.grid @grid0(shape = 8x8x8)
shard.sharding @grid0 split_axes = [[0]] halo_sizes = [1] sharded_dims_offsets = [1] : !shard.sharding

// CHECK: 'shard.sharding' op halo sizes and shard offsets are mutually exclusive
