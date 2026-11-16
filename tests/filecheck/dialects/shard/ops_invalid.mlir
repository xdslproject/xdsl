// RUN: xdsl-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect | filecheck %s

shard.grid @grid0(shape = [])

// CHECK: 'shard.grid' op rank of grid is expected to be a positive integer 

// -----


shard.grid @grid0(shape = 8x8x8)
shard.sharding @grid0 split_axes = [[0]] halo_sizes = [1] sharded_dims_offsets = [1] : !shard.sharding

// CHECK: 'shard.sharding' op halo sizes and shard offsets are mutually exclusive

// -----

%t = "test.op"() : () -> tensor<2x2xi8>
%0 = shard.all_gather %t on @grid0 gather_axis = 0 : tensor<2x2xi8> -> tensor<2x2xi8>

// CHECK: Undefined required grid symbol "grid0".

// -----

shard.grid @grid0(shape = 2x4)
%t = "test.op"() : () -> tensor<2x2xi8>
%0 = shard.all_gather %t on @grid0 grid_axes = [0, 0] gather_axis = 0 : tensor<2x2xi8> -> tensor<8x2xi8>

// CHECK: Grid axes contains duplicate elements.

// -----

shard.grid @grid0(shape = 2x4)
%t = "test.op"() : () -> tensor<2x2xi8>
%0 = shard.all_gather %t on @grid0 grid_axes = [2] gather_axis = 0 : tensor<2x2xi8> -> tensor<8x2xi8>

// CHECK: 0-based grid axis index 2 is out of bounds. The referenced grid "grid0" is of rank 2.

// -----

shard.grid @grid0(shape = 2x4)
%t = "test.op"() : () -> tensor<2x2xi8>
%0 = shard.all_gather %t on @grid0 grid_axes = [1] gather_axis = 2 : tensor<2x2xi8> -> tensor<2x2xi8>

// CHECK: Gather axis 2 is out of bounds [0, 2).

// -----

shard.grid @grid0(shape = 2x4)
%t = "test.op"() : () -> tensor<2x2xi8>
%0 = shard.all_gather %t on @grid0 grid_axes = [1] gather_axis = 0 : tensor<2x2xi8> -> tensor<4x2xi8>

// CHECK: Dimension size mismatch for result axis 0. Expected 8, but got 4.

// -----

shard.grid @grid0(shape = 2x?)
%t = "test.op"() : () -> tensor<2x2xi8>
%0 = shard.all_gather %t on @grid0 grid_axes = [1] gather_axis = 0 : tensor<2x2xi8> -> tensor<4x2xi8>

// CHECK: Dimension size mismatch for result axis 0. Expected dynamic, but got 4.

// -----

shard.grid @grid0(shape = 2x4)
%t = "test.op"() : () -> tensor<2x2xi8>
%0 = shard.all_gather %t on @grid0 grid_axes = [1] gather_axis = 0 : tensor<2x2xi8> -> tensor<8x2xi16>

// CHECK: 'shard.all_gather' op requires the same element type for all operands and results

// -----

shard.grid @grid0(shape = 2x4)
%t = "test.op"() : () -> tensor<2x2xi8>
%0 = shard.all_gather %t on @grid0 grid_axes = [1] gather_axis = 0 : tensor<2x2xi8> -> tensor<8x2x1xi8>

// CHECK: 'shard.all_gather' op result type has different rank than operands

// -----

shard.grid @grid0(shape = 2x4)
%t = "test.op"() : () -> tensor<i8>
%0 = shard.all_gather %t on @grid0 grid_axes = [1] gather_axis = 0 : tensor<i8> -> tensor<i8>

// CHECK: incorrect length for range variable

// -----

shard.grid @grid0(shape = 2x4)
%t = "test.op"() : () -> tensor<3x2xi8>
%0 = shard.all_slice %t on @grid0 grid_axes = [0] slice_axis = 0 : tensor<3x2xi8> -> tensor<1x2xi8>

// CHECK: Operand dimension size 3 is not divisible by collective device group size 2 for tensor axis 0.

// -----

shard.grid @grid0(shape = 2x4)
%t = "test.op"() : () -> tensor<4x2xi8>
%0 = shard.all_slice %t on @grid0 grid_axes = [0] slice_axis = 0 : tensor<4x2xi8> -> tensor<2x1xi8>

// CHECK: Dimension size mismatch for result axis 1. Expected 2, but got 1.

// -----

shard.grid @grid0(shape = 2x4)
%t = "test.op"() : () -> tensor<4x2xi8>
%0 = shard.reduce_scatter %t on @grid0 grid_axes = [0, 1] scatter_axis = 0 : tensor<4x2xi8> -> tensor<2x2xi64>

// CHECK: Operand dimension size 4 is not divisible by collective device group size 8 for tensor axis 0.

// -----

shard.grid @grid0(shape = 2x4)
%t = "test.op"() : () -> tensor<4x2xi8>
%0 = shard.reduce_scatter %t on @grid0 grid_axes = [0] scatter_axis = 0 : tensor<4x2xi8> -> tensor<1x2xi64>

// CHECK: Dimension size mismatch for result axis 0. Expected 2, but got 1.

// -----

shard.grid @grid0(shape = 2x4)
%t = "test.op"() : () -> tensor<4x2xi8>
%0 = shard.reduce_scatter %t on @grid0 grid_axes = [0] scatter_axis = 2 : tensor<4x2xi8> -> tensor<2x2xi8>

// CHECK: Tensor axis 2 is out of bounds [0, 2).
