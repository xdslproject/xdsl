// RUN: xdsl-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect | filecheck %s

mesh.mesh @mesh0(shape = [])

// CHECK: 'mesh.mesh' op rank of mesh is expected to be a positive integer 

// -----


mesh.mesh @mesh0(shape = 8x8x8)
mesh.sharding @mesh0 split_axes = [[0]] halo_sizes = [1] sharded_dims_offsets = [1] : !mesh.sharding

// CHECK: 'mesh.sharding' cannot use both `halo_sizes` and `sharded_dims_offsets`
