// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

func.func @omp_ordered(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) {
    "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = 0 : i64}> ({
        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
        ^bb0(%arg7: i32):
            omp.yield
        }) : (i32, i32, i32) -> ()
        "omp.terminator"() : () -> ()
    }) : () -> ()
    return
}

// CHECK: Operation does not verify: omp.wsloop is not a LoopWrapper: has 2 ops, expected 1

// -----

func.func @omp_ordered(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) {
    "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = 0 : i64}> ({
        "omp.terminator"() : () -> ()
    }) : () -> ()
    return
}

// CHECK: omp.wsloop is not a LoopWrapper: should have a single operation which is either another LoopWrapper or omp.loop_nest

// -----

func.func @omp_wsloop_block(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) {
    "omp.wsloop"(%arg0, %arg1, %arg3) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 2, 1, 0>, ordered = 0 : i64}> ({
    ^block_(%a: i32):
        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
        ^bb0(%arg7: i32):
            omp.yield
        }) : (i32, i32, i32) -> ()
    }) : (i32, i32, i64) -> ()
    return
}

// CHECK: omp.wsloop expected to have at least 3 block argument(s), got 1

// -----

func.func @omp_parallel_block(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) {
  "omp.parallel"(%arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 1, 2>}> ({
    ^block_(%a: i32):
    "omp.terminator"() : () -> ()
  }) : (i32, i32, i32) -> ()
  return
}

// CHECK: omp.parallel expected to have at least 3 block argument(s), got 1

// -----

func.func @omp_target_block(%host: i32, %inr1: i32, %inr2: i32, %map: memref<1xf32>, %arg4: i64, %arg5: i64, %arg6: i64) {
  %map1 = "omp.map.info"(%map) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = #omp<clause_map_flags to>, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target"(%host, %inr1, %inr2, %map1, %arg4, %arg5, %arg6) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 3, 0>}> ({
    ^bb0(%a: i32, %b: i32):
    "omp.terminator"() : () -> ()
  }) : (i32, i32, i32, memref<1xf32>, i64, i64, i64) -> ()
  return
}

// CHECK: omp.target expected to have at least 7 block argument(s), got 2

// -----

func.func @omp_simd(%ub: index, %lb: index, %step: index,%p1: f32, %r1: memref<1xf32>) {
  "omp.simd"(%p1, %r1) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 1, 1>}> ({
  ^bb0():
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb0(%iter: index):
      omp.yield
    }) : (index, index, index) -> ()

  }) : (f32, memref<1xf32>) -> ()
  func.return
}

// CHECK: omp.simd expected to have at least 2 block argument(s), got 0

// -----

func.func @omp_target_data(%dev: i64, %if: i1, %m: memref<1xf32>, %d1: memref<1xf64>, %d2: memref<1xi32>) {
  %m1 = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = #omp<clause_map_flags to>, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target_data"(%dev, %if, %m1, %d1, %d2, %d2) <{operandSegmentSizes = array<i32: 1, 1, 1, 2, 1>}> ({
  ^bb0(%0: memref<1xi64>, %1: memref<1xi64>):
    "omp.terminator"() : () -> ()
  }) : (i64, i1, memref<1xf32>, memref<1xf64>, memref<1xi32>, memref<1xi32>) -> ()
  func.return
}


// CHECK: omp.target_data expected to have at least 3 block argument(s), got 2

// -----

func.func @omp_simd_aligned(%ub: index, %lb: index, %step: index, %a1: memref<1xi32>, %a2: memref<10xf32>) {
  "omp.simd"(%a1, %a2) <{operandSegmentSizes = array<i32: 2, 0, 0, 0, 0, 0, 0>, alignments = [64, 8, 16]}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb0(%iter: index):
      omp.yield
    }) : (index, index, index) -> ()

  }) : (memref<1xi32>, memref<10xf32>) -> ()
  func.return
}

// CHECK: integer 2 expected from int variable 'ALIGN_COUNT', but got 3

// -----

func.func @omp_simd_linear(%ub: index, %lb: index, %step: index, %l1: memref<1xi32>, %lstep1: i32, %lstep2: i32) {
  "omp.simd"(%l1, %lstep1, %lstep2) <{linear_var_types = [i32], operandSegmentSizes = array<i32: 0, 0, 1, 2, 0, 0, 0>}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb0(%iter: index):
      omp.yield
    }) : (index, index, index) -> ()

  }) : (memref<1xi32>, i32, i32) -> ()
  func.return
}

// CHECK: integer 1 expected from int variable 'LINEAR_COUNT', but got 2

// -----

func.func @omp_simd_simdlen(%ub: index, %lb: index, %step: index) {
  "omp.simd"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, simdlen=8, safelen=2}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb0(%iter: index):
      omp.yield
    }) : (index, index, index) -> ()

  }) : () -> ()
  func.return
}

// CHECK: `safelen` must be greater than or equal to `simdlen`

// -----

func.func @omp_simd_simdlen_0(%ub: index, %lb: index, %step: index) {
  "omp.simd"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, simdlen=0, safelen=2}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb0(%iter: index):
      omp.yield
    }) : (index, index, index) -> ()

  }) : () -> ()
  func.return
}

// CHECK: expected integer >= 1, got 0

// -----

func.func @omp_target_data_no_map_info(%m: memref<1xf32>) {
  "omp.target_data"(%m) <{operandSegmentSizes = array<i32: 0, 0, 1, 0, 0>}> ({
    "omp.terminator"() : () -> ()
  }) : (memref<1xf32>) -> ()
  func.return
}

// CHECK: All mapped operands of omp.target_data must be results of a omp.map.info

// -----

func.func @omp_target_data_delete(%m: memref<1xf32>) {
  %m1 = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type =  #omp<clause_map_flags del>, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target_data"(%m1) <{operandSegmentSizes = array<i32: 0, 0, 1, 0, 0>}> ({
    "omp.terminator"() : () -> ()
  }) : (memref<1xf32>) -> ()
  func.return
}

// CHECK: Cannot have map_type DELETE in omp.target_data

// -----

func.func @omp_target_data_to_and_delete(%dev: i64, %if: i1, %m: memref<1xf32>, %d1: memref<1xf32>, %d2: memref<1xf32>) {
  %m1 = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type =  #omp<clause_map_flags to|del>, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target_data"(%m1) <{operandSegmentSizes = array<i32: 0, 0, 1, 0, 0>}> ({
  ^bb0(%0: memref<1xf32>, %1: memref<1xf32>, %2: memref<1xf32>):
    "omp.terminator"() : () -> ()
  }) : (memref<1xf32>) -> ()
  func.return
}

// CHECK: Cannot have map_type DELETE in omp.target_data

// -----

func.func @omp_target_enter_data_from(%m: memref<1xf32>) {
  %from = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type =  #omp<clause_map_flags from>, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
"omp.target_enter_data"(%from) <{operandSegmentSizes = array<i32: 0, 0, 0, 1>}> : (memref<1xf32>) -> ()
  func.return
}

// CHECK: Cannot have map_type FROM in omp.target_enter_data

// -----

func.func @omp_target_enter_data_delete(%m: memref<1xf32>) {
  %del = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type =  #omp<clause_map_flags del>, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
"omp.target_enter_data"(%del) <{operandSegmentSizes = array<i32: 0, 0, 0, 1>}> : (memref<1xf32>) -> ()
  func.return
}

// CHECK: Cannot have map_type DELETE in omp.target_enter_data

// -----

func.func @omp_target_exit_data_to(%m: memref<1xf32>) {
  %to = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = #omp<clause_map_flags to>, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target_exit_data"(%to) <{operandSegmentSizes = array<i32: 0, 0, 0, 1>}> : (memref<1xf32>) -> ()
  func.return
}

// CHECK: Cannot have map_type TO in omp.target_exit_data

// -----

func.func @omp_target_update_del(%m: memref<1xf32>) {
  %del = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = #omp<clause_map_flags del>, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target_update"(%del) <{operandSegmentSizes = array<i32: 0, 0, 0, 1>}> : (memref<1xf32>) -> ()
  func.return
}

// CHECK: Cannot have map_type DELETE in omp.target_update

// -----

func.func @omp_target_update_to_from_same_map(%m: memref<1xf32>) {
  %tofrom = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = #omp<clause_map_flags to|from>, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target_update"(%tofrom) <{operandSegmentSizes = array<i32: 0, 0, 0, 1>}> : (memref<1xf32>) -> ()
  func.return
}

// CHECK: omp.target_update expected to have exactly one of TO or FROM as map_type

// -----

func.func @omp_target_update_to_from_same_operand(%m: memref<1xf32>) {
  %to = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = #omp<clause_map_flags to>, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  %from = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = #omp<clause_map_flags from>, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target_update"(%to, %from) <{operandSegmentSizes = array<i32: 0, 0, 0, 2>}> : (memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// CHECK: omp.target_update expected to have exactly one of TO or FROM as map_type

// -----

func.func @wsloopop_linear(%ub: index, %lb: index, %step: index, %l1: memref<1xi32>, %lstep1: i32, %lstep2: i32) {
  "omp.wsloop"(%l1, %lstep1, %lstep2) <{operandSegmentSizes = array<i32: 0, 0, 1, 2, 0, 0, 0>, linear_var_types = [i32]}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb0(%iter: index):
      omp.yield
    }) : (index, index, index) -> ()

  }) : (memref<1xi32>, i32, i32) -> ()
  func.return
}

// -----

// CHECK: integer 1 expected from int variable 'LINEAR_COUNT', but got 2

func.func @wsloopop_ordered(%ub: index, %lb: index, %step: index, %l1: memref<1xi32>, %lstep1: i32, %lstep2: i32) {
  "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = -1 : i64}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb0(%iter: index):
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
  ^bb0(%r1_arg: i32):
    cf.br  ^bb1
  ^bb1():
    omp.yield(%r1_arg : i32)
  }, {
  }, {
  }, {
  }, {
  }, {
  }) : () -> ()

  func.return
}

// CHECK: omp.declare_reduction should have at most 1 block in alloc_region

// -----

func.func @omp_distribute_chunk_operand(%lb: index, %ub: index, %step: index, %chunk: i64) {
  "omp.distribute"(%chunk) <{operandSegmentSizes = array<i32: 0, 0, 1, 0>}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb1(%iter: index):
      omp.yield
    }) : (index, index, index) -> ()
  }) : (i64) -> ()
  func.return
}

// CHECK: omp.distribute should have either both dist_schedule_static and dist_schedule_chunk_size, or neither.

// -----

func.func @omp_distribute_chunk_attr(%lb: index, %ub: index, %step: index) {
  "omp.distribute"() <{dist_schedule_static, operandSegmentSizes = array<i32: 0, 0, 0, 0>}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb1(%iter: index):
      omp.yield
    }) : (index, index, index) -> ()
  }) : () -> ()
  func.return
}

// CHECK: omp.distribute should have either both dist_schedule_static and dist_schedule_chunk_size, or neither.

// -----

func.func @omp_taskwait_depend_count(%d1: memref<1xi32>) {
  "omp.taskwait"(%d1) <{depend_kinds = [#omp<clause_task_depend (taskdependin)>, #omp<clause_task_depend (taskdependout)>]}> : (memref<1xi32>) -> ()
  func.return
}

// CHECK: integer 1 expected from int variable 'DEP_COUNT', but got 2

// -----

func.func @omp_single_copyprivate(%cp: memref<1xi32>) {
  "omp.single"(%cp) <{operandSegmentSizes = array<i32: 0, 0, 1, 0>}> ({
    "omp.terminator"() : () -> ()
  }) : (memref<1xi32>) -> ()
  func.return
}

// CHECK: omp.single inconsistent number of copyprivate vars (1) and functions (0)

// -----

func.func @omp_task_depend_count(%dep: memref<1xi32>) {
  "omp.task"(%dep) <{depend_kinds = [#omp<clause_task_depend (taskdependin)>, #omp<clause_task_depend (taskdependout)>], operandSegmentSizes = array<i32: 0, 0, 1, 0, 0, 0, 0, 0, 0>}> ({
    "omp.terminator"() : () -> ()
  }) : (memref<1xi32>) -> ()
  func.return
}

// CHECK: integer 1 expected from int variable 'DEP_COUNT', but got 2

// -----

func.func @omp_task_in_reduction_syms(%ir: memref<1xi32>) {
  "omp.task"(%ir) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 1, 0, 0, 0>}> ({
  ^bb0(%ir_arg: memref<1xi32>):
    "omp.terminator"() : () -> ()
  }) : (memref<1xi32>) -> ()
  func.return
}

// CHECK: omp.task expected as many in_reduction symbol references as in_reduction variables

// -----

func.func @omp_task_in_reduction_byref(%ir: memref<1xi32>) {
  "omp.task"(%ir) <{in_reduction_byref = array<i1: false, true>, in_reduction_syms = [@r1], operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 1, 0, 0, 0>}> ({
  ^bb0(%ir_arg: memref<1xi32>):
    "omp.terminator"() : () -> ()
  }) : (memref<1xi32>) -> ()
  func.return
}

// CHECK: omp.task expected as many in_reduction byref flags as in_reduction variables

// -----

func.func @omp_task_block_args(%p1: i32) {
  "omp.task"(%p1) <{private_syms = [@p1], operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 1, 0>}> ({
    "omp.terminator"() : () -> ()
  }) : (i32) -> ()
  func.return
}

// CHECK: omp.task expected to have at least 1 block argument(s), got 0

// -----

func.func @omp_taskgroup_task_reduction_syms(%tr: memref<1xi32>) {
  "omp.taskgroup"(%tr) <{operandSegmentSizes = array<i32: 0, 0, 1>}> ({
  ^bb0(%tr_arg: memref<1xi32>):
    "omp.terminator"() : () -> ()
  }) : (memref<1xi32>) -> ()
  func.return
}

// CHECK: omp.taskgroup expected as many task_reduction symbol references as task_reduction variables

// -----

func.func @omp_taskgroup_task_reduction_byref(%tr: memref<1xi32>) {
  "omp.taskgroup"(%tr) <{task_reduction_byref = array<i1: false, true>, task_reduction_syms = [@r1], operandSegmentSizes = array<i32: 0, 0, 1>}> ({
  ^bb0(%tr_arg: memref<1xi32>):
    "omp.terminator"() : () -> ()
  }) : (memref<1xi32>) -> ()
  func.return
}

// CHECK: omp.taskgroup expected as many task_reduction byref flags as task_reduction variables

// -----

func.func @omp_taskgroup_block_args(%tr: memref<1xi32>) {
  "omp.taskgroup"(%tr) <{task_reduction_syms = [@r1], operandSegmentSizes = array<i32: 0, 0, 1>}> ({
    "omp.terminator"() : () -> ()
  }) : (memref<1xi32>) -> ()
  func.return
}

// CHECK: omp.taskgroup expected to have at least 1 block argument(s), got 0

// -----

func.func @omp_taskloop_grainsize_num_tasks(%lb: index, %ub: index, %step: index, %g: i64) {
  "omp.taskloop"(%g, %g) <{operandSegmentSizes = array<i32: 0, 0, 0, 1, 0, 0, 1, 0, 0, 0>}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb1(%iv: index):
      omp.yield
    }) : (index, index, index) -> ()
  }) : (i64, i64) -> ()
  func.return
}

// CHECK: the grainsize clause and num_tasks clause are mutually exclusive and may not appear on the same taskloop directive

// -----

func.func @omp_taskloop_reduction_byref(%lb: index, %ub: index, %step: index, %r: memref<1xi32>) {
  "omp.taskloop"(%r) <{reduction_syms = [@r1], reduction_byref = array<i1: false, false>, operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 1>}> ({
  ^bb0(%r_arg: memref<1xi32>):
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb1(%iv: index):
      omp.yield
    }) : (index, index, index) -> ()
  }) : (memref<1xi32>) -> ()
  func.return
}

// CHECK: omp.taskloop expected as many reduction byref flags as reduction variables

// -----

func.func @omp_taskloop_not_loop_wrapper() {
  "omp.taskloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
    "omp.terminator"() : () -> ()
  }) : () -> ()
  func.return
}

// CHECK: omp.taskloop is not a LoopWrapper: should have a single operation which is either another LoopWrapper or omp.loop_nest

// -----

func.func @omp_taskloop_in_reduction_syms(%lb: index, %ub: index, %step: index, %ir: memref<1xi32>) {
  "omp.taskloop"(%ir) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 1, 0, 0, 0, 0>}> ({
  ^bb0(%ir_arg: memref<1xi32>):
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb1(%iv: index):
      omp.yield
    }) : (index, index, index) -> ()
  }) : (memref<1xi32>) -> ()
  func.return
}

// CHECK: omp.taskloop expected as many in_reduction symbol references as in_reduction variables

// -----

func.func @omp_taskloop_in_reduction_byref(%lb: index, %ub: index, %step: index, %ir: memref<1xi32>) {
  "omp.taskloop"(%ir) <{in_reduction_byref = array<i1: false, true>, in_reduction_syms = [@r1], operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 1, 0, 0, 0, 0>}> ({
  ^bb0(%ir_arg: memref<1xi32>):
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb1(%iv: index):
      omp.yield
    }) : (index, index, index) -> ()
  }) : (memref<1xi32>) -> ()
  func.return
}

// CHECK: omp.taskloop expected as many in_reduction byref flags as in_reduction variables

// -----

func.func @omp_taskloop_reduction_syms(%lb: index, %ub: index, %step: index, %r: memref<1xi32>) {
  "omp.taskloop"(%r) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 1>}> ({
  ^bb0(%r_arg: memref<1xi32>):
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^bb1(%iv: index):
      omp.yield
    }) : (index, index, index) -> ()
  }) : (memref<1xi32>) -> ()
  func.return
}

// CHECK: omp.taskloop expected as many reduction symbol references as reduction variables
