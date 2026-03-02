// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

func.func @omp_ordered(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i64, %arg4 : i64, %arg5 : i64, %arg6 : i64) {
    "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = 0 : i64}> ({
        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
        ^bb0(%arg7 : i32):
            omp.yield
        }) : (i32, i32, i32) -> ()
        "omp.terminator"() : () -> ()
    }) : () -> ()
    return
}

// CHECK: Operation does not verify: omp.wsloop is not a LoopWrapper: has 2 ops, expected 1

// -----

func.func @omp_ordered(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i64, %arg4 : i64, %arg5 : i64, %arg6 : i64) {
    "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = 0 : i64}> ({
        "omp.terminator"() : () -> ()
    }) : () -> ()
    return
}

// CHECK: omp.wsloop is not a LoopWrapper: should have a single operation which is either another LoopWrapper or omp.loop_nest

// -----

func.func @omp_wsloop_block(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i64, %arg4 : i64, %arg5 : i64, %arg6 : i64) {
    "omp.wsloop"(%arg0, %arg1, %arg3) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 2, 1, 0>, ordered = 0 : i64}> ({
    ^block_(%a : i32):
        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
        ^bb0(%arg7 : i32):
            omp.yield
        }) : (i32, i32, i32) -> ()
    }) : (i32, i32, i64) -> ()
    return
}

// CHECK: omp.wsloop expected to have at least 3 block argument(s), got 1

// -----

func.func @omp_parallel_block(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i64, %arg4 : i64, %arg5 : i64, %arg6 : i64) {
  "omp.parallel"(%arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 1, 2>}> ({
    ^block_(%a : i32):
    "omp.terminator"() : () -> ()
  }) : (i32, i32, i32) -> ()
  return
}

// CHECK: omp.parallel expected to have at least 3 block argument(s), got 1

// -----

func.func @omp_target_block(%host : i32, %inr1 : i32, %inr2 : i32, %map : memref<1xf32>, %arg4 : i64, %arg5 : i64, %arg6 : i64) {
  %map1 = "omp.map.info"(%map) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 1 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target"(%host, %inr1, %inr2, %map1, %arg4, %arg5, %arg6) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 3, 0>}> ({
    ^bb0(%a : i32, %b : i32):
    "omp.terminator"() : () -> ()
  }) : (i32, i32, i32, memref<1xf32>, i64, i64, i64) -> ()
  return
}

// CHECK: omp.target expected to have at least 7 block argument(s), got 2

// -----

func.func @omp_simd(%ub : index, %lb : index, %step : index,%p1 : f32, %r1 : memref<1xf32>) {
  "omp.simd"(%p1, %r1) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 1, 1>}> ({
  ^bb0():
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb0(%iter : index):
      omp.yield
    }) : (index, index, index) -> ()

  }) : (f32, memref<1xf32>) -> ()
  func.return
}

// CHECK: omp.simd expected to have at least 2 block argument(s), got 0

// -----

func.func @omp_target_data(%dev : i64, %if : i1, %m : memref<1xf32>, %d1 : memref<1xf64>, %d2 : memref<1xi32>) {
  %m1 = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 1 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target_data"(%dev, %if, %m1, %d1, %d2, %d2) <{operandSegmentSizes = array<i32: 1, 1, 1, 2, 1>}> ({
  ^bb0(%0 : memref<1xi64>, %1 : memref<1xi64>):
    "omp.terminator"() : () -> ()
  }) : (i64, i1, memref<1xf32>, memref<1xf64>, memref<1xi32>, memref<1xi32>) -> ()
  func.return
}


// CHECK: omp.target_data expected to have at least 3 block argument(s), got 2

// -----

func.func @omp_simd_aligned(%ub : index, %lb : index, %step : index, %a1 : memref<1xi32>, %a2 : memref<10xf32>) {
  "omp.simd"(%a1, %a2) <{operandSegmentSizes = array<i32: 2, 0, 0, 0, 0, 0, 0>, alignments = [64, 8, 16]}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb0(%iter : index):
      omp.yield
    }) : (index, index, index) -> ()

  }) : (memref<1xi32>, memref<10xf32>) -> ()
  func.return
}

// CHECK: integer 2 expected from int variable 'ALIGN_COUNT', but got 3

// -----

func.func @omp_simd_linear(%ub : index, %lb : index, %step : index, %l1 : memref<1xi32>, %lstep1 : i32, %lstep2 : i32) {
  "omp.simd"(%l1, %lstep1, %lstep2) <{operandSegmentSizes = array<i32: 0, 0, 1, 2, 0, 0, 0>}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb0(%iter : index):
      omp.yield
    }) : (index, index, index) -> ()

  }) : (memref<1xi32>, i32, i32) -> ()
  func.return
}

// CHECK: integer 1 expected from int variable 'LINEAR_COUNT', but got 2

// -----

func.func @omp_simd_simdlen(%ub : index, %lb : index, %step : index) {
  "omp.simd"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, simdlen=8, safelen=2}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb0(%iter : index):
      omp.yield
    }) : (index, index, index) -> ()

  }) : () -> ()
  func.return
}

// CHECK: `safelen` must be greater than or equal to `simdlen`

// -----

func.func @omp_simd_simdlen_0(%ub : index, %lb : index, %step : index) {
  "omp.simd"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, simdlen=0, safelen=2}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb0(%iter : index):
      omp.yield
    }) : (index, index, index) -> ()

  }) : () -> ()
  func.return
}

// CHECK: expected integer >= 1, got 0

// -----

func.func @omp_target_data_no_map_info(%m : memref<1xf32>) {
  "omp.target_data"(%m) <{operandSegmentSizes = array<i32: 0, 0, 1, 0, 0>}> ({
    "omp.terminator"() : () -> ()
  }) : (memref<1xf32>) -> ()
  func.return
}

// CHECK: All mapped operands of omp.target_data must be results of a omp.map.info

// -----

func.func @omp_target_data_delete(%m : memref<1xf32>) {
  %m1 = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 0x08 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target_data"(%m1) <{operandSegmentSizes = array<i32: 0, 0, 1, 0, 0>}> ({
    "omp.terminator"() : () -> ()
  }) : (memref<1xf32>) -> ()
  func.return
}

// CHECK: Cannot have map_type DELETE in omp.target_data

// -----

func.func @omp_target_data_to_and_delete(%dev : i64, %if : i1, %m : memref<1xf32>, %d1 : memref<1xf32>, %d2 : memref<1xf32>) {
  %m1 = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 0x9 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target_data"(%m1) <{operandSegmentSizes = array<i32: 0, 0, 1, 0, 0>}> ({
  ^bb0(%0 : memref<1xf32>, %1 : memref<1xf32>, %2 : memref<1xf32>):
    "omp.terminator"() : () -> ()
  }) : (memref<1xf32>) -> ()
  func.return
}

// CHECK: Cannot have map_type DELETE in omp.target_data

// -----

func.func @omp_target_enter_data_from(%m : memref<1xf32>) {
  %from = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 0x2 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
"omp.target_enter_data"(%from) <{operandSegmentSizes = array<i32: 0, 0, 0, 1>}> : (memref<1xf32>) -> ()
  func.return
}

// CHECK: Cannot have map_type FROM in omp.target_enter_data

// -----

func.func @omp_target_enter_data_delete(%m : memref<1xf32>) {
  %del = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 0x8 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
"omp.target_enter_data"(%del) <{operandSegmentSizes = array<i32: 0, 0, 0, 1>}> : (memref<1xf32>) -> ()
  func.return
}

// CHECK: Cannot have map_type DELETE in omp.target_enter_data

// -----

func.func @omp_target_exit_data_to(%m : memref<1xf32>) {
  %to = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 0x1 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target_exit_data"(%to) <{operandSegmentSizes = array<i32: 0, 0, 0, 1>}> : (memref<1xf32>) -> ()
  func.return
}

// CHECK: Cannot have map_type TO in omp.target_exit_data

// -----

func.func @omp_target_update_del(%m : memref<1xf32>) {
  %del = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 0x8 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target_update"(%del) <{operandSegmentSizes = array<i32: 0, 0, 0, 1>}> : (memref<1xf32>) -> ()
  func.return
}

// CHECK: Cannot have map_type DELETE in omp.target_update

// -----

func.func @omp_target_update_to_from_same_map(%m : memref<1xf32>) {
  %tofrom = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 0x3 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target_update"(%tofrom) <{operandSegmentSizes = array<i32: 0, 0, 0, 1>}> : (memref<1xf32>) -> ()
  func.return
}

// CHECK: omp.target_update expected to have exactly one of TO or FROM as map_type

// -----

func.func @omp_target_update_to_from_same_operand(%m : memref<1xf32>) {
  %to = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 0x1 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  %from = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 0x2 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target_update"(%to, %from) <{operandSegmentSizes = array<i32: 0, 0, 0, 2>}> : (memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// CHECK: omp.target_update expected to have exactly one of TO or FROM as map_type

// -----

func.func @wsloopop_linear(%ub : index, %lb : index, %step : index, %l1 : memref<1xi32>, %lstep1 : i32, %lstep2 : i32) {
  "omp.wsloop"(%l1, %lstep1, %lstep2) <{operandSegmentSizes = array<i32: 0, 0, 1, 2, 0, 0, 0>}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb0(%iter : index):
      omp.yield
    }) : (index, index, index) -> ()

  }) : (memref<1xi32>, i32, i32) -> ()
  func.return
}

// -----

// CHECK: integer 1 expected from int variable 'LINEAR_COUNT', but got 2

func.func @wsloopop_ordered(%ub : index, %lb : index, %step : index, %l1 : memref<1xi32>, %lstep1 : i32, %lstep2 : i32) {
  "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = -1 : i64}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb0(%iter : index):
      omp.yield
    }) : (index, index, index) -> ()

  }) : () -> ()
  func.return
}

// CHECK: expected integer >= 0, got -1

// -----

func.func @yield_parent() {
  "test.op"() ({
    omp.yield
  }) : () -> ()

  func.return
}

// CHECK: 'omp.yield' expects parent op to be one of 'omp.loop_nest', 'omp.private', 'omp.declare_reduction'

// -----

func.func @reduction_too_many_blocks() {
  "omp.declare_reduction"() <{sym_name = "r1", type = i32}> ({
  ^bb0(%r1_arg : i32):
    cf.br  ^bb1
  ^bb1():
    omp.yield(%r1_arg : i32)
  }, {
  }, {
  }, {
  }, {
  }) : () -> ()

  func.return
}

// CHECK: omp.declare_reduction should have at most 1 block in alloc_region

// -----

func.func @omp_distribute_chunk_operand(%lb : index, %ub : index, %step : index, %chunk : i64) {
  "omp.distribute"(%chunk) <{operandSegmentSizes = array<i32: 0, 0, 1, 0>}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb1(%iter : index):
      omp.yield
    }) : (index, index, index) -> ()
  }) : (i64) -> ()
  func.return
}

// CHECK: omp.distribute should have either both dist_schedule_static and dist_schedule_chunk_size, or neither.

// -----

func.func @omp_distribute_chunk_attr(%lb : index, %ub : index, %step : index) {
  "omp.distribute"() <{dist_schedule_static, operandSegmentSizes = array<i32: 0, 0, 0, 0>}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb1(%iter : index):
      omp.yield
    }) : (index, index, index) -> ()
  }) : () -> ()
  func.return
}

// CHECK: omp.distribute should have either both dist_schedule_static and dist_schedule_chunk_size, or neither.
