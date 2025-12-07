// RUN: xdsl-opt -p gpu-map-parallel-loops --split-input-file %s | filecheck %s

func.func @parallel_loop(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index) {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  %2 = arith.constant 4 : index
  "scf.parallel"(%arg0, %arg1, %arg2, %arg3, %2, %2) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%arg4 : index, %arg5 : index):
    "scf.parallel"(%0, %0, %2, %2, %1, %1) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
    ^bb1(%arg6 : index, %arg7 : index):
      "test.op"() : () -> ()
      scf.reduce
    }) : (index, index, index, index, index, index) -> ()
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  func.return
}

// CHECK:         func @parallel_loop(
// CHECK:           scf.parallel
// CHECK:             scf.parallel
// CHECK:      {mapping = [#gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// CHECK-SAME:             #gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK:      {mapping = [#gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// CHECK-SAME:             #gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}

// -----

func.func @parallel_loop_4d(%arg0_1 : index, %arg1_1 : index, %arg2_1 : index, %arg3_1 : index) {
  %3 = arith.constant 0 : index
  %4 = arith.constant 1 : index
  %5 = arith.constant 4 : index
  "scf.parallel"(%3, %3, %3, %3, %arg0_1, %arg1_1, %arg2_1, %arg3_1, %5, %5, %5, %5) <{operandSegmentSizes = array<i32: 4, 4, 4, 0>}> ({
  ^bb2(%arg4_1 : index, %arg5_1 : index, %arg6_1 : index, %arg7_1 : index):
    "scf.parallel"(%3, %3, %3, %3, %5, %5, %5, %5, %4, %4, %4, %4) <{operandSegmentSizes = array<i32: 4, 4, 4, 0>}> ({
    ^bb3(%arg8 : index, %arg9 : index, %arg10 : index, %arg11 : index):
      "scf.parallel"(%3, %3, %3, %3, %5, %5, %5, %5, %4, %4, %4, %4) <{operandSegmentSizes = array<i32: 4, 4, 4, 0>}> ({
      ^bb4(%arg12 : index, %arg13 : index, %arg14 : index, %arg15 : index):
        "test.op"() : () -> ()
        scf.reduce
      }) : (index, index, index, index, index, index, index, index, index, index, index, index) -> ()
      scf.reduce
    }) : (index, index, index, index, index, index, index, index, index, index, index, index) -> ()
    scf.reduce
  }) : (index, index, index, index, index, index, index, index, index, index, index, index) -> ()
  func.return
}

// CHECK:         func @parallel_loop_4d(
// CHECK:           scf.parallel
// CHECK:             scf.parallel
// CHECK:               scf.parallel
// CHECK:      {mapping = [#gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// CHECK-SAME:             #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// CHECK-SAME:             #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// CHECK-SAME:             #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK:      {mapping = [#gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// CHECK-SAME:             #gpu.loop_dim_map<processor = thread_z, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// CHECK-SAME:             #gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// CHECK-SAME:             #gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK:      {mapping = [#gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// CHECK-SAME:             #gpu.loop_dim_map<processor = block_z, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// CHECK-SAME:             #gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
// CHECK-SAME:             #gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
