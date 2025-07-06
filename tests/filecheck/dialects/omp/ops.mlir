// RUN: XDSL_ROUNDTRIP

builtin.module {
  func.func @omp_parallel(%arg0 : memref<1xi32>, %arg1 : i1, %arg2 : i32) {
    "omp.parallel"(%arg0, %arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>, "proc_bind_kind" = #omp.procbindkind spread}> ({
      "omp.parallel"(%arg0, %arg0, %arg2) <{operandSegmentSizes = array<i32: 1, 1, 0, 1, 0, 0>}> ({
        "omp.terminator"() : () -> ()
      }) : (memref<1xi32>, memref<1xi32>, i32) -> ()
      "omp.parallel"(%arg0, %arg0, %arg1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>}> ({
        "omp.terminator"() : () -> ()
      }) : (memref<1xi32>, memref<1xi32>, i1) -> ()
      "omp.parallel"(%arg1, %arg2) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 0>}> ({
        "omp.terminator"() : () -> ()
      }) : (i1, i32) -> ()
      "omp.terminator"() : () -> ()
    }) : (memref<1xi32>, memref<1xi32>, i1, i32) -> ()
    "omp.parallel"(%arg0, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>}> ({
      "omp.terminator"() : () -> ()
    }) : (memref<1xi32>, memref<1xi32>) -> ()
    func.return
  }
  func.func @omp_ordered(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i64, %arg4 : i64, %arg5 : i64, %arg6 : i64) {
    "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = 0 : i64}> ({
      "omp.loop_nest"(%arg0, %arg1, %arg2) ({
      ^0(%arg7 : i32):
        omp.yield
      }) : (i32, i32, i32) -> ()
    }) : () -> ()
    "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = 1 : i64}> ({
      "omp.loop_nest"(%arg0, %arg1, %arg2) ({
      ^1(%arg7_1 : i32):
        omp.yield
      }) : (i32, i32, i32) -> ()
    }) : () -> ()
    "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = 2 : i64}> ({
      "omp.loop_nest"(%arg0, %arg1, %arg2) ({
      ^2(%arg7_2 : i32):
        omp.yield
      }) : (i32, i32, i32) -> ()
    }) : () -> ()
    func.return
  }
  func.func @omp_wsloop(%arg0_1 : index, %arg1_1 : index, %arg2_1 : index, %arg3_1 : memref<1xi32>, %arg4_1 : i32, %arg5_1 : i32) {
    "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = 1 : i64}> ({
      "omp.loop_nest"(%arg0_1, %arg1_1, %arg2_1) ({
      ^3(%arg6_1 : index):
        omp.yield
      }) : (index, index, index) -> ()
    }) : () -> ()
    "omp.wsloop"(%arg3_1, %arg4_1) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 0, 0>, "schedule_kind" = #omp<schedulekind static>}> ({
      "omp.loop_nest"(%arg0_1, %arg1_1, %arg2_1) ({
      ^4(%arg6_2 : index):
        omp.yield
      }) : (index, index, index) -> ()
    }) : (memref<1xi32>, i32) -> ()
    "omp.wsloop"(%arg3_1, %arg3_1, %arg4_1, %arg4_1) <{operandSegmentSizes = array<i32: 0, 0, 2, 2, 0, 0, 0>, "schedule_kind" = #omp<schedulekind static>}> ({
      "omp.loop_nest"(%arg0_1, %arg1_1, %arg2_1) ({
      ^5(%arg6_3 : index):
        omp.yield
      }) : (index, index, index) -> ()
    }) : (memref<1xi32>, memref<1xi32>, i32, i32) -> ()
    "omp.wsloop"(%arg3_1, %arg4_1, %arg5_1) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 0, 1>, ordered = 2 : i64, "schedule_kind" = #omp<schedulekind dynamic>}> ({
      "omp.loop_nest"(%arg0_1, %arg1_1, %arg2_1) ({
      ^6(%arg6_4 : index):
        omp.yield
      }) : (index, index, index) -> ()
    }) : (memref<1xi32>, i32, i32) -> ()
    "omp.wsloop"() <{"nowait", operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, "schedule_kind" = #omp<schedulekind auto>}> ({
      "omp.loop_nest"(%arg0_1, %arg1_1, %arg2_1) ({
      ^7(%arg6_5 : index):
        omp.yield
      }) : (index, index, index) -> ()
    }) : () -> ()
    func.return
  }
  func.func @omp_wsloop_pretty(%arg0_2 : index, %arg1_2 : index, %arg2_2 : index, %arg3_2 : memref<1xi32>, %arg4_2 : i32, %arg5_2 : i32, %arg6_6 : i16) {
    "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = 2 : i64}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^8(%arg7_3 : index):
        omp.yield
      }) : (index, index, index) -> ()
    }) : () -> ()
    "omp.wsloop"(%arg3_2, %arg4_2) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 0, 0>, "schedule_kind" = #omp<schedulekind static>}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^9(%arg7_4 : index):
        omp.yield
      }) : (index, index, index) -> ()
    }) : (memref<1xi32>, i32) -> ()
    "omp.wsloop"(%arg3_2, %arg4_2, %arg5_2) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 0, 1>, ordered = 2 : i64, "schedule_kind" = #omp<schedulekind static>}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^10(%arg7_5 : index):
        omp.yield
      }) : (index, index, index) -> ()
    }) : (memref<1xi32>, i32, i32) -> ()
    "omp.wsloop"(%arg3_2, %arg4_2, %arg5_2) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 0, 1>, ordered = 2 : i64, "schedule_mod" = #omp<sched_mod nonmonotonic>, "schedule_kind" = #omp<schedulekind dynamic>}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^11(%arg7_6 : index):
        omp.yield
      }) : (index, index, index) -> ()
    }) : (memref<1xi32>, i32, i32) -> ()
    "omp.wsloop"(%arg3_2, %arg4_2, %arg6_6) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 0, 1>, ordered = 2 : i64, "schedule_mod" = #omp<sched_mod monotonic>, "schedule_kind" = #omp<schedulekind dynamic>}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^12(%arg7_7 : index):
        omp.yield
      }) : (index, index, index) -> ()
    }) : (memref<1xi32>, i32, i16) -> ()
    "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^13(%arg7_8 : index):
        omp.yield
      }) : (index, index, index) -> ()
    }) : () -> ()
    "omp.wsloop"() <{"inclusive", operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^14(%arg7_9 : index):
        omp.yield
      }) : (index, index, index) -> ()
    }) : () -> ()
    "omp.wsloop"() <{"nowait", operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^15(%arg7_10 : index):
        omp.yield
      }) : (index, index, index) -> ()
    }) : () -> ()
    "omp.wsloop"() <{"nowait", operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, order = #omp<orderkind concurrent>}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^16(%arg7_11 : index):
        omp.yield
      }) : (index, index, index) -> ()
    }) : () -> ()
    func.return
  }
  func.func @omp_target(%dep : memref<6xf32>, %dev : i64, %host : i32, %if : i1, %p1 : memref<10xi8>, %p2 : f64, %tlimit : i32) {
    "omp.target"(%dep, %dev, %host, %if, %p1, %p2, %tlimit) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 2, 1>, in_reduction_syms = [@rsym1, @rsym2], private_syms = [@psym1, @psym2], nowait, bare, depend_kinds = [#omp<clause_task_depend (taskdependinout)>]}> ({
      "omp.terminator"() : () -> ()
    }) : (memref<6xf32>, i64, i32, i1, memref<10xi8>, f64, i32) -> ()
    func.return
  }
  func.func @omp_map_info_bounds(%lb : index, %ub : index, %ptr : memref<1xf32>, %ptr_ptr : memref<1xmemref<1xf32>>) {
    %bounds = "omp.map.bounds"(%lb, %ub) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (index, index) -> !omp.map_bounds_ty
    %ptr_info = "omp.map.info"(%ptr, %ptr_ptr, %bounds) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, var_type = memref<1xf32>, map_type = 1 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>, name = "ptr_info"}> : (memref<1xf32>, memref<1xmemref<1xf32>>, !omp.map_bounds_ty) -> memref<1xf32>
    %ptr_ptr_info = "omp.map.info"(%ptr_ptr, %ptr_info, %bounds) <{operandSegmentSizes = array<i32: 1, 0, 1, 1>, map_type = 1 : ui64, map_capture_type = #omp<variable_capture_kind (ByCopy)>, var_type = memref<1xmemref<1xf32>>}> : (memref<1xmemref<1xf32>>, memref<1xf32>, !omp.map_bounds_ty) -> memref<1xmemref<1xf32>>
    func.return
  }
  func.func @omp_simd(%ub : index, %lb : index, %step : index, %if : i1, %nt : memref<1xi32>, %p1 : f32, %r1 : memref<1xf32>) {
    "omp.simd"(%if, %nt, %p1, %r1) <{operandSegmentSizes = array<i32: 0, 1, 0, 0, 1, 1, 1>, order = #omp<orderkind concurrent>, private_syms=[@p1], reduction_syms=[@r1], simdlen = 64 : i64, safelen = 128 : i64}> ({
    ^0(%1 : f32, %2 : f32):
      "omp.loop_nest"(%lb, %ub, %step) ({
      ^0(%iter : index):
        omp.yield
      }) : (index, index, index) -> ()

    }) : (i1, memref<1xi32>, f32, memref<1xf32>) -> ()
    func.return
  }

  func.func @omp_simd_aligned(%ub : index, %lb : index, %step : index, %a1 : memref<1xi32>, %a2 : memref<10xf32>) {
    "omp.simd"(%a1, %a2) <{operandSegmentSizes = array<i32: 2, 0, 0, 0, 0, 0, 0>, alignments = [64, 8]}> ({
      "omp.loop_nest"(%lb, %ub, %step) ({
      ^0(%iter : index):
        omp.yield
      }) : (index, index, index) -> ()

    }) : (memref<1xi32>, memref<10xf32>) -> ()
    func.return
  }

  func.func @omp_simd_linear(%ub : index, %lb : index, %step : index, %l1 : memref<1xi32>, %lstep : i32) {
    "omp.simd"(%l1, %lstep) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 0, 0>}> ({
      "omp.loop_nest"(%lb, %ub, %step) ({
      ^0(%iter : index):
        omp.yield
      }) : (index, index, index) -> ()

    }) : (memref<1xi32>, i32) -> ()
    func.return
  }
  func.func @omp_target_data(%dev : i64, %if : i1, %m : memref<1xf32>, %d1 : memref<1xf32>, %d2 : memref<1xf32>) {
    %m1 = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 1 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
    "omp.target_data"(%dev, %if, %m1, %d1, %d2, %d2) <{operandSegmentSizes = array<i32: 1, 1, 1, 2, 1>}> ({
    ^0(%0 : memref<1xf32>, %1 : memref<1xf32>, %2 : memref<1xf32>):
      "omp.terminator"() : () -> ()
    }) : (i64, i1, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
    func.return
  }

  func.func @omp_target_data_task(%dep: memref<1xi32>, %dev : i32, %if : i1, %m1 : memref<1xf32>, %m2 : memref<1xf32>, %m3: memref<1xf32>) {
    %to = "omp.map.info"  (%m1) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 0x01 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
    %from = "omp.map.info"(%m2) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 0x02 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
    %del = "omp.map.info" (%m3) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 0x08 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
    "omp.target_enter_data"(%dep, %dev, %if, %to)         <{operandSegmentSizes = array<i32: 1, 1, 1, 1>, "nowait", depend_kinds=[#omp<clause_task_depend(taskdependin)>]}>: (memref<1xi32>, i32, i1, memref<1xf32>) -> ()
    "omp.target_update"    (%dep, %dev, %if, %to, %from)  <{operandSegmentSizes = array<i32: 1, 1, 1, 2>, "nowait", depend_kinds=[#omp<clause_task_depend(taskdependin)>]}>: (memref<1xi32>, i32, i1, memref<1xf32>, memref<1xf32>) -> ()
    "omp.target_exit_data" (%dep, %dev, %if, %from, %del) <{operandSegmentSizes = array<i32: 1, 1, 1, 2>, "nowait", depend_kinds=[#omp<clause_task_depend(taskdependin)>]}>: (memref<1xi32>, i32, i1, memref<1xf32>, memref<1xf32>) -> ()
    func.return
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @omp_parallel(%arg0 : memref<1xi32>, %arg1 : i1, %arg2 : i32) {
// CHECK-NEXT:      "omp.parallel"(%arg0, %arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>, proc_bind_kind = #omp<procbindkind spread>}> ({
// CHECK-NEXT:        "omp.parallel"(%arg0, %arg0, %arg2) <{operandSegmentSizes = array<i32: 1, 1, 0, 1, 0, 0>}> ({
// CHECK-NEXT:          "omp.terminator"() : () -> ()
// CHECK-NEXT:        }) : (memref<1xi32>, memref<1xi32>, i32) -> ()
// CHECK-NEXT:        "omp.parallel"(%arg0, %arg0, %arg1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>}> ({
// CHECK-NEXT:          "omp.terminator"() : () -> ()
// CHECK-NEXT:        }) : (memref<1xi32>, memref<1xi32>, i1) -> ()
// CHECK-NEXT:        "omp.parallel"(%arg1, %arg2) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 0>}> ({
// CHECK-NEXT:          "omp.terminator"() : () -> ()
// CHECK-NEXT:        }) : (i1, i32) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, memref<1xi32>, i1, i32) -> ()
// CHECK-NEXT:      "omp.parallel"(%arg0, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, memref<1xi32>) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @omp_ordered(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i64, %arg4 : i64, %arg5 : i64, %arg6 : i64) {
// CHECK-NEXT:      "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = 0 : i64}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^0(%arg7 : i32):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (i32, i32, i32) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = 1 : i64}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^1(%arg7_1 : i32):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (i32, i32, i32) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = 2 : i64}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^2(%arg7_2 : i32):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (i32, i32, i32) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @omp_wsloop(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : memref<1xi32>, %arg4 : i32, %arg5 : i32) {
// CHECK-NEXT:      "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = 1 : i64}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^0(%arg6 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg3, %arg4) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 0, 0>, schedule_kind = #omp<schedulekind static>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^1(%arg6_1 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg3, %arg3, %arg4, %arg4) <{operandSegmentSizes = array<i32: 0, 0, 2, 2, 0, 0, 0>, schedule_kind = #omp<schedulekind static>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^2(%arg6_2 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, memref<1xi32>, i32, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg3, %arg4, %arg5) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 0, 1>, ordered = 2 : i64, schedule_kind = #omp<schedulekind dynamic>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^3(%arg6_3 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, i32, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"() <{nowait, operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, schedule_kind = #omp<schedulekind auto>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^4(%arg6_4 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @omp_wsloop_pretty(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : memref<1xi32>, %arg4 : i32, %arg5 : i32, %arg6 : i16) {
// CHECK-NEXT:      "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = 2 : i64}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^0(%arg7 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg3, %arg4) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 0, 0>, schedule_kind = #omp<schedulekind static>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^1(%arg7_1 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg3, %arg4, %arg5) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 0, 1>, ordered = 2 : i64, schedule_kind = #omp<schedulekind static>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^2(%arg7_2 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, i32, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg3, %arg4, %arg5) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 0, 1>, ordered = 2 : i64, schedule_mod = #omp<sched_mod nonmonotonic>, schedule_kind = #omp<schedulekind dynamic>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^3(%arg7_3 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, i32, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg3, %arg4, %arg6) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 0, 1>, ordered = 2 : i64, schedule_mod = #omp<sched_mod monotonic>, schedule_kind = #omp<schedulekind dynamic>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^4(%arg7_4 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, i32, i16) -> ()
// CHECK-NEXT:      "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^5(%arg7_5 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.wsloop"() <{inclusive, operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^6(%arg7_6 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.wsloop"() <{nowait, operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^7(%arg7_7 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.wsloop"() <{nowait, operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, order = #omp<orderkind concurrent>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^8(%arg7_8 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @omp_target(%dep : memref<6xf32>, %dev : i64, %host : i32, %if : i1, %p1 : memref<10xi8>, %p2 : f64, %tlimit : i32) {
// CHECK-NEXT:      "omp.target"(%dep, %dev, %host, %if, %p1, %p2, %tlimit) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 2, 1>, in_reduction_syms = [@rsym1, @rsym2], private_syms = [@psym1, @psym2], nowait, bare, depend_kinds = [#omp<clause_task_depend (taskdependinout)>]}> ({
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : (memref<6xf32>, i64, i32, i1, memref<10xi8>, f64, i32) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @omp_map_info_bounds(%lb : index, %ub : index, %ptr : memref<1xf32>, %ptr_ptr : memref<1xmemref<1xf32>>) {
// CHECK-NEXT:      %bounds = "omp.map.bounds"(%lb, %ub) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, stride_in_bytes = false}> : (index, index) -> !omp.map_bounds_ty
// CHECK-NEXT:      %ptr_info = "omp.map.info"(%ptr, %ptr_ptr, %bounds) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, var_type = memref<1xf32>, map_type = 1 : ui64, map_capture_type = #omp<variable_capture_kind (ByCopy)>, name = "ptr_info"}> : (memref<1xf32>, memref<1xmemref<1xf32>>, !omp.map_bounds_ty) -> memref<1xf32>
// CHECK-NEXT:      %ptr_ptr_info = "omp.map.info"(%ptr_ptr, %ptr_info, %bounds) <{operandSegmentSizes = array<i32: 1, 0, 1, 1>, map_type = 1 : ui64, map_capture_type = #omp<variable_capture_kind (ByCopy)>, var_type = memref<1xmemref<1xf32>>}> : (memref<1xmemref<1xf32>>, memref<1xf32>, !omp.map_bounds_ty) -> memref<1xmemref<1xf32>>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @omp_simd(%ub : index, %lb : index, %step : index, %if : i1, %nt : memref<1xi32>, %p1 : f32, %r1 : memref<1xf32>) {
// CHECK-NEXT:      "omp.simd"(%if, %nt, %p1, %r1) <{operandSegmentSizes = array<i32: 0, 1, 0, 0, 1, 1, 1>, order = #omp<orderkind concurrent>, private_syms = [@p1], reduction_syms = [@r1], simdlen = 64 : i64, safelen = 128 : i64}> ({
// CHECK-NEXT:      ^0(%0 : f32, %1 : f32):
// CHECK-NEXT:        "omp.loop_nest"(%lb, %ub, %step) ({
// CHECK-NEXT:        ^1(%iter : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:      }) : (i1, memref<1xi32>, f32, memref<1xf32>) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @omp_simd_aligned(%ub : index, %lb : index, %step : index, %a1 : memref<1xi32>, %a2 : memref<10xf32>) {
// CHECK-NEXT:      "omp.simd"(%a1, %a2) <{operandSegmentSizes = array<i32: 2, 0, 0, 0, 0, 0, 0>, alignments = [64 : i64, 8 : i64]}> ({
// CHECK-NEXT:        "omp.loop_nest"(%lb, %ub, %step) ({
// CHECK-NEXT:        ^0(%iter : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, memref<10xf32>) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @omp_simd_linear(%ub : index, %lb : index, %step : index, %l1 : memref<1xi32>, %lstep : i32) {
// CHECK-NEXT:      "omp.simd"(%l1, %lstep) <{operandSegmentSizes = array<i32: 0, 0, 1, 1, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%lb, %ub, %step) ({
// CHECK-NEXT:        ^0(%iter : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, i32) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @omp_target_data(%dev : i64, %if : i1, %m : memref<1xf32>, %d1 : memref<1xf32>, %d2 : memref<1xf32>) {
// CHECK-NEXT:      %m1 = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 1 : ui64, map_capture_type = #omp<variable_capture_kind (ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
// CHECK-NEXT:      "omp.target_data"(%dev, %if, %m1, %d1, %d2, %d2) <{operandSegmentSizes = array<i32: 1, 1, 1, 2, 1>}> ({
// CHECK-NEXT:      ^0(%0 : memref<1xf32>, %1 : memref<1xf32>, %2 : memref<1xf32>):
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : (i64, i1, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @omp_target_data_task(%dep : memref<1xi32>, %dev : i32, %if : i1, %m1 : memref<1xf32>, %m2 : memref<1xf32>, %m3 : memref<1xf32>) {
// CHECK-NEXT:      %to = "omp.map.info"(%m1) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 1 : ui64, map_capture_type = #omp<variable_capture_kind (ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
// CHECK-NEXT:      %from = "omp.map.info"(%m2) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 2 : ui64, map_capture_type = #omp<variable_capture_kind (ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
// CHECK-NEXT:      %del = "omp.map.info"(%m3) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 8 : ui64, map_capture_type = #omp<variable_capture_kind (ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
// CHECK-NEXT:      "omp.target_enter_data"(%dep, %dev, %if, %to) <{operandSegmentSizes = array<i32: 1, 1, 1, 1>, nowait, depend_kinds = [#omp<clause_task_depend (taskdependin)>]}> : (memref<1xi32>, i32, i1, memref<1xf32>) -> ()
// CHECK-NEXT:      "omp.target_update"(%dep, %dev, %if, %to, %from) <{operandSegmentSizes = array<i32: 1, 1, 1, 2>, nowait, depend_kinds = [#omp<clause_task_depend (taskdependin)>]}> : (memref<1xi32>, i32, i1, memref<1xf32>, memref<1xf32>) -> ()
// CHECK-NEXT:      "omp.target_exit_data"(%dep, %dev, %if, %from, %del) <{operandSegmentSizes = array<i32: 1, 1, 1, 2>, nowait, depend_kinds = [#omp<clause_task_depend (taskdependin)>]}> : (memref<1xi32>, i32, i1, memref<1xf32>, memref<1xf32>) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
