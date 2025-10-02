// RUN: XDSL_ROUNDTRIP

mesh.mesh @mesh0(shape = 2x2x4)
mesh.mesh @mesh1(shape = 4x?)
mesh.mesh @mesh2(shape = ?x4)
mesh.mesh @mesh3(shape = ?x?)
mesh.mesh @mesh4(shape = 3)
mesh.mesh @mesh5(shape = ?)

%0 = mesh.sharding @mesh0 split_axes = [[]] : !mesh.sharding
%1 = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
%2 = mesh.sharding @mesh0 split_axes = [[1]] : !mesh.sharding
%3 = mesh.sharding @mesh1 split_axes = [[], [0]] : !mesh.sharding
%4 = mesh.sharding @mesh3 split_axes = [[0], [], [1]] : !mesh.sharding
%5 = mesh.sharding @mesh3 split_axes = [[0]] partial = max[1] : !mesh.sharding
%6 = mesh.sharding @mesh3 split_axes = [[0]] partial = min[1] : !mesh.sharding
%7 = mesh.sharding @mesh3 split_axes = [[0]] partial = generic[1] : !mesh.sharding
%8 = mesh.sharding @mesh3 split_axes = [[0]] partial = sum[1, 2] : !mesh.sharding
%9 = mesh.sharding @mesh4 split_axes = [[0]] halo_sizes = [1, 4] : !mesh.sharding

%10 = arith.constant 3 : i64
%11 = mesh.sharding @mesh4 split_axes = [[0]] halo_sizes = [4, %10] : !mesh.sharding
%12 = mesh.sharding @mesh4 split_axes = [[0]] sharded_dims_offsets = [0, 1, 4, 6] : !mesh.sharding
%13 = mesh.sharding @mesh4 split_axes = [[0]] sharded_dims_offsets = [0, 2, %10, 5] : !mesh.sharding

// Collective communications
%t = "test.op"() : () -> tensor<2x2xi8>
%14 = mesh.broadcast %t on @mesh0 mesh_axes = [0] root = [0] : (tensor<2x2xi8>) -> tensor<2x2xi8>
%15 = mesh.gather %t on @mesh0 mesh_axes = [1] gather_axis = 1 root = [1] : (tensor<2x2xi8>) -> tensor<2x4xi8>
%16 = mesh.scatter %t on @mesh0 mesh_axes = [0] scatter_axis = 0 root = [1] : (tensor<2x2xi8>) -> tensor<1x2xi8>
%17 = mesh.recv %t on @mesh0 mesh_axes = [0, 2] source = [0, 1] : (tensor<2x2xi8>) -> tensor<2x2xi8>
%18 = mesh.send %t on @mesh0 mesh_axes = [1] destination = [1] : (tensor<2x2xi8>) -> tensor<2x2xi8>
%19 = mesh.shift %t on @mesh0 mesh_axes = [1] shift_axis = 1 offset = 2 rotate : tensor<2x2xi8> -> tensor<2x2xi8>


// CHECK:      mesh.mesh @mesh0(shape = 2x2x4)
// CHECK-NEXT: mesh.mesh @mesh1(shape = 4x?)
// CHECK-NEXT: mesh.mesh @mesh2(shape = ?x4)
// CHECK-NEXT: mesh.mesh @mesh3(shape = ?x?)
// CHECK-NEXT: mesh.mesh @mesh4(shape = 3)
// CHECK-NEXT: mesh.mesh @mesh5(shape = ?)

// CHECK-NEXT: %0 = mesh.sharding @mesh0 split_axes = [[]] : !mesh.sharding
// CHECK-NEXT: %1 = mesh.sharding @mesh0 split_axes = [[0]] : !mesh.sharding
// CHECK-NEXT: %2 = mesh.sharding @mesh0 split_axes = [[1]] : !mesh.sharding
// CHECK-NEXT: %3 = mesh.sharding @mesh1 split_axes = [[], [0]] : !mesh.sharding
// CHECK-NEXT: %4 = mesh.sharding @mesh3 split_axes = [[0], [], [1]] : !mesh.sharding
// CHECK-NEXT: %5 = mesh.sharding @mesh3 split_axes = [[0]] partial = max [1] : !mesh.sharding
// CHECK-NEXT: %6 = mesh.sharding @mesh3 split_axes = [[0]] partial = min [1] : !mesh.sharding
// CHECK-NEXT: %7 = mesh.sharding @mesh3 split_axes = [[0]] partial = generic [1] : !mesh.sharding
// CHECK-NEXT: %8 = mesh.sharding @mesh3 split_axes = [[0]] partial = sum [1, 2] : !mesh.sharding
// CHECK-NEXT: %9 = mesh.sharding @mesh4 split_axes = [[0]] halo_sizes = [1, 4] : !mesh.sharding
// CHECK-NEXT: %10 = arith.constant 3 : i64
// CHECK-NEXT: %11 = mesh.sharding @mesh4 split_axes = [[0]] halo_sizes = [4, %10] : !mesh.sharding
// CHECK-NEXT: %12 = mesh.sharding @mesh4 split_axes = [[0]] sharded_dims_offsets = [0, 1, 4, 6] : !mesh.sharding
// CHECK-NEXT: %13 = mesh.sharding @mesh4 split_axes = [[0]] sharded_dims_offsets = [0, 2, %10, 5] : !mesh.sharding
// CHECK-NEXT: %t = "test.op"() : () -> tensor<2x2xi8>
// CHECK-NEXT: %14 = mesh.broadcast %t on @mesh0 mesh_axes = [0] root = [0] : (tensor<2x2xi8>) -> tensor<2x2xi8>
// CHECK-NEXT: %15 = mesh.gather %t on @mesh0 mesh_axes = [1] gather_axis = 1 root = [1] : (tensor<2x2xi8>) -> tensor<2x4xi8>
// CHECK-NEXT: %16 = mesh.scatter %t on @mesh0 mesh_axes = [0] scatter_axis = 0 root = [1] : (tensor<2x2xi8>) -> tensor<1x2xi8>
// CHECK-NEXT: %17 = mesh.recv %t on @mesh0 mesh_axes = [0, 2] source = [0, 1] : (tensor<2x2xi8>) -> tensor<2x2xi8>
// CHECK-NEXT: %18 = mesh.send %t on @mesh0 mesh_axes = [1] destination = [1] : (tensor<2x2xi8>) -> tensor<2x2xi8>
// CHECK-NEXT: %19 = mesh.shift %t on @mesh0 mesh_axes = [1] shift_axis = 1 offset = 2 rotate : tensor<2x2xi8> -> tensor<2x2xi8>
