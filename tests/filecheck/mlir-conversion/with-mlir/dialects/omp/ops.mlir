// RUN: xdsl-opt %s --print-op-generic | mlir-opt --mlir-print-op-generic | xdsl-opt | filecheck %s

builtin.module {
  func.func @omp_parallel(%arg0 : memref<1xi32>, %arg1 : i1, %arg2 : i32) {
    "omp.parallel"(%arg1, %arg2, %arg0, %arg0) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 1, 0, 0>, "proc_bind_val" = #omp.procbindkind spread}> ({
      "omp.parallel"(%arg2, %arg0, %arg0) <{"operandSegmentSizes" = array<i32: 0, 1, 1, 1, 0, 0>}> ({
        "omp.terminator"() : () -> ()
      }) : (i32, memref<1xi32>, memref<1xi32>) -> ()
      "omp.parallel"(%arg1, %arg0, %arg0) <{"operandSegmentSizes" = array<i32: 1, 0, 1, 1, 0, 0>}> ({
        "omp.terminator"() : () -> ()
      }) : (i1, memref<1xi32>, memref<1xi32>) -> ()
      "omp.parallel"(%arg1, %arg2) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 0, 0, 0>}> ({
        "omp.terminator"() : () -> ()
      }) : (i1, i32) -> ()
      "omp.terminator"() : () -> ()
    }) : (i1, i32, memref<1xi32>, memref<1xi32>) -> ()
    "omp.parallel"(%arg0, %arg0) <{"operandSegmentSizes" = array<i32: 0, 0, 1, 1, 0, 0>}> ({
      "omp.terminator"() : () -> ()
    }) : (memref<1xi32>, memref<1xi32>) -> ()
    func.return
  }
  func.func @omp_ordered(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i64, %arg4 : i64, %arg5 : i64, %arg6 : i64) {
    "omp.wsloop"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0>, "ordered_val" = 0 : i64}> ({
      "omp.loop_nest"(%arg0, %arg1, %arg2) ({
      ^0(%arg7 : i32):
        omp.yield
      }) : (i32, i32, i32) -> ()
      "omp.terminator"() : () -> ()
    }) : () -> ()
    "omp.wsloop"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0>, "ordered_val" = 1 : i64}> ({
      "omp.loop_nest"(%arg0, %arg1, %arg2) ({
      ^1(%arg7_1 : i32):
        omp.yield
      }) : (i32, i32, i32) -> ()
      "omp.terminator"() : () -> ()
    }) : () -> ()
    "omp.wsloop"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0>, "ordered_val" = 2 : i64}> ({
      "omp.loop_nest"(%arg0, %arg1, %arg2) ({
      ^2(%arg7_2 : i32):
        omp.yield
      }) : (i32, i32, i32) -> ()
      "omp.terminator"() : () -> ()
    }) : () -> ()
    func.return
  }
  func.func @omp_wsloop(%arg0_1 : index, %arg1_1 : index, %arg2_1 : index, %arg3_1 : memref<1xi32>, %arg4_1 : i32, %arg5_1 : i32) {
    "omp.wsloop"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0>, "ordered_val" = 1 : i64}> ({
      "omp.loop_nest"(%arg0_1, %arg1_1, %arg2_1) ({
      ^3(%arg6_1 : index):
        omp.yield
      }) : (index, index, index) -> ()
      "omp.terminator"() : () -> ()
    }) : () -> ()
    "omp.wsloop"(%arg3_1, %arg4_1) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 0>, "schedule_val" = #omp<schedulekind static>}> ({
      "omp.loop_nest"(%arg0_1, %arg1_1, %arg2_1) ({
      ^4(%arg6_2 : index):
        omp.yield
      }) : (index, index, index) -> ()
      "omp.terminator"() : () -> ()
    }) : (memref<1xi32>, i32) -> ()
    "omp.wsloop"(%arg3_1, %arg3_1, %arg4_1, %arg4_1) <{"operandSegmentSizes" = array<i32: 2, 2, 0, 0>, "schedule_val" = #omp<schedulekind static>}> ({
      "omp.loop_nest"(%arg0_1, %arg1_1, %arg2_1) ({
      ^5(%arg6_3 : index):
        omp.yield
      }) : (index, index, index) -> ()
      "omp.terminator"() : () -> ()
    }) : (memref<1xi32>, memref<1xi32>, i32, i32) -> ()
    "omp.wsloop"(%arg3_1, %arg4_1, %arg5_1) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 1>, "ordered_val" = 2 : i64, "schedule_val" = #omp<schedulekind dynamic>}> ({
      "omp.loop_nest"(%arg0_1, %arg1_1, %arg2_1) ({
      ^6(%arg6_4 : index):
        omp.yield
      }) : (index, index, index) -> ()
      "omp.terminator"() : () -> ()
    }) : (memref<1xi32>, i32, i32) -> ()
    "omp.wsloop"() <{"nowait", "operandSegmentSizes" = array<i32: 0, 0, 0, 0>, "schedule_val" = #omp<schedulekind auto>}> ({
      "omp.loop_nest"(%arg0_1, %arg1_1, %arg2_1) ({
      ^7(%arg6_5 : index):
        omp.yield
      }) : (index, index, index) -> ()
      "omp.terminator"() : () -> ()
    }) : () -> ()
    func.return
  }
  func.func @omp_wsloop_pretty(%arg0_2 : index, %arg1_2 : index, %arg2_2 : index, %arg3_2 : memref<1xi32>, %arg4_2 : i32, %arg5_2 : i32, %arg6_6 : i16) {
    "omp.wsloop"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0>, "ordered_val" = 2 : i64}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^8(%arg7_3 : index):
        omp.yield
      }) : (index, index, index) -> ()
      "omp.terminator"() : () -> ()
    }) : () -> ()
    "omp.wsloop"(%arg3_2, %arg4_2) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 0>, "schedule_val" = #omp<schedulekind static>}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^9(%arg7_4 : index):
        omp.yield
      }) : (index, index, index) -> ()
      "omp.terminator"() : () -> ()
    }) : (memref<1xi32>, i32) -> ()
    "omp.wsloop"(%arg3_2, %arg4_2, %arg5_2) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 1>, "ordered_val" = 2 : i64, "schedule_val" = #omp<schedulekind static>}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^10(%arg7_5 : index):
        omp.yield
      }) : (index, index, index) -> ()
      "omp.terminator"() : () -> ()
    }) : (memref<1xi32>, i32, i32) -> ()
    "omp.wsloop"(%arg3_2, %arg4_2, %arg5_2) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 1>, "ordered_val" = 2 : i64, "schedule_modifier" = #omp<sched_mod nonmonotonic>, "schedule_val" = #omp<schedulekind dynamic>}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^11(%arg7_6 : index):
        omp.yield
      }) : (index, index, index) -> ()
      "omp.terminator"() : () -> ()
    }) : (memref<1xi32>, i32, i32) -> ()
    "omp.wsloop"(%arg3_2, %arg4_2, %arg6_6) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 1>, "ordered_val" = 2 : i64, "schedule_modifier" = #omp<sched_mod monotonic>, "schedule_val" = #omp<schedulekind dynamic>}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^12(%arg7_7 : index):
        omp.yield
      }) : (index, index, index) -> ()
      "omp.terminator"() : () -> ()
    }) : (memref<1xi32>, i32, i16) -> ()
    "omp.wsloop"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0>}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^13(%arg7_8 : index):
        omp.yield
      }) : (index, index, index) -> ()
      "omp.terminator"() : () -> ()
    }) : () -> ()
    "omp.wsloop"() <{"inclusive", "operandSegmentSizes" = array<i32: 0, 0, 0, 0>}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^14(%arg7_9 : index):
        omp.yield
      }) : (index, index, index) -> ()
      "omp.terminator"() : () -> ()
    }) : () -> ()
    "omp.wsloop"() <{"nowait", "operandSegmentSizes" = array<i32: 0, 0, 0, 0>}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^15(%arg7_10 : index):
        omp.yield
      }) : (index, index, index) -> ()
      "omp.terminator"() : () -> ()
    }) : () -> ()
    "omp.wsloop"() <{"nowait", "operandSegmentSizes" = array<i32: 0, 0, 0, 0>, "order_val" = #omp<orderkind concurrent>}> ({
      "omp.loop_nest"(%arg0_2, %arg1_2, %arg2_2) ({
      ^16(%arg7_11 : index):
        omp.yield
      }) : (index, index, index) -> ()
      "omp.terminator"() : () -> ()
    }) : () -> ()
    func.return
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @omp_parallel(%arg37 : memref<1xi32>, %arg38 : i1, %arg39 : i32) {
// CHECK-NEXT:      "omp.parallel"(%arg38, %arg39, %arg37, %arg37) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 1, 0, 0>, "proc_bind_val" = #omp<procbindkind spread>}> ({
// CHECK-NEXT:        "omp.parallel"(%arg39, %arg37, %arg37) <{"operandSegmentSizes" = array<i32: 0, 1, 1, 1, 0, 0>}> ({
// CHECK-NEXT:          "omp.terminator"() : () -> ()
// CHECK-NEXT:        }) : (i32, memref<1xi32>, memref<1xi32>) -> ()
// CHECK-NEXT:        "omp.parallel"(%arg38, %arg37, %arg37) <{"operandSegmentSizes" = array<i32: 1, 0, 1, 1, 0, 0>}> ({
// CHECK-NEXT:          "omp.terminator"() : () -> ()
// CHECK-NEXT:        }) : (i1, memref<1xi32>, memref<1xi32>) -> ()
// CHECK-NEXT:        "omp.parallel"(%arg38, %arg39) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 0, 0, 0>}> ({
// CHECK-NEXT:          "omp.terminator"() : () -> ()
// CHECK-NEXT:        }) : (i1, i32) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : (i1, i32, memref<1xi32>, memref<1xi32>) -> ()
// CHECK-NEXT:      "omp.parallel"(%arg37, %arg37) <{"operandSegmentSizes" = array<i32: 0, 0, 1, 1, 0, 0>}> ({
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, memref<1xi32>) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @omp_ordered(%arg27 : i32, %arg28 : i32, %arg29 : i32, %arg30 : i64, %arg31 : i64, %arg32 : i64, %arg33 : i64) {
// CHECK-NEXT:      "omp.wsloop"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0>, "ordered_val" = 0 : i64}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg27, %arg28, %arg29) ({
// CHECK-NEXT:        ^0(%arg36 : i32):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (i32, i32, i32) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.wsloop"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0>, "ordered_val" = 1 : i64}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg27, %arg28, %arg29) ({
// CHECK-NEXT:        ^1(%arg35 : i32):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (i32, i32, i32) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.wsloop"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0>, "ordered_val" = 2 : i64}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg27, %arg28, %arg29) ({
// CHECK-NEXT:        ^2(%arg34 : i32):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (i32, i32, i32) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @omp_wsloop(%arg16 : index, %arg17 : index, %arg18 : index, %arg19 : memref<1xi32>, %arg20 : i32, %arg21 : i32) {
// CHECK-NEXT:      "omp.wsloop"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0>, "ordered_val" = 1 : i64}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg16, %arg17, %arg18) ({
// CHECK-NEXT:        ^0(%arg26 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg19, %arg20) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 0>, "schedule_val" = #omp<schedulekind static>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg16, %arg17, %arg18) ({
// CHECK-NEXT:        ^1(%arg25 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg19, %arg19, %arg20, %arg20) <{"operandSegmentSizes" = array<i32: 2, 2, 0, 0>, "schedule_val" = #omp<schedulekind static>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg16, %arg17, %arg18) ({
// CHECK-NEXT:        ^2(%arg24 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, memref<1xi32>, i32, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg19, %arg20, %arg21) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 1>, "ordered_val" = 2 : i64, "schedule_val" = #omp<schedulekind dynamic>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg16, %arg17, %arg18) ({
// CHECK-NEXT:        ^3(%arg23 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, i32, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"() <{"nowait", "operandSegmentSizes" = array<i32: 0, 0, 0, 0>, "schedule_val" = #omp<schedulekind auto>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg16, %arg17, %arg18) ({
// CHECK-NEXT:        ^4(%arg22 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @omp_wsloop_pretty(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : memref<1xi32>, %arg4 : i32, %arg5 : i32, %arg6 : i16) {
// CHECK-NEXT:      "omp.wsloop"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0>, "ordered_val" = 2 : i64}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^0(%arg15 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg3, %arg4) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 0>, "schedule_val" = #omp<schedulekind static>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^1(%arg14 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg3, %arg4, %arg5) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 1>, "ordered_val" = 2 : i64, "schedule_val" = #omp<schedulekind static>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^2(%arg13 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, i32, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg3, %arg4, %arg5) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 1>, "ordered_val" = 2 : i64, "schedule_modifier" = #omp<sched_mod nonmonotonic>, "schedule_val" = #omp<schedulekind dynamic>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^3(%arg12 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, i32, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg3, %arg4, %arg6) <{"operandSegmentSizes" = array<i32: 1, 1, 0, 1>, "ordered_val" = 2 : i64, "schedule_modifier" = #omp<sched_mod monotonic>, "schedule_val" = #omp<schedulekind dynamic>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^4(%arg11 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : (memref<1xi32>, i32, i16) -> ()
// CHECK-NEXT:      "omp.wsloop"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^5(%arg10 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.wsloop"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^6(%arg9 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.wsloop"() <{"nowait", "operandSegmentSizes" = array<i32: 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^7(%arg8 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.wsloop"() <{"nowait", "operandSegmentSizes" = array<i32: 0, 0, 0, 0>, "order_val" = #omp<orderkind concurrent>}> ({
// CHECK-NEXT:        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
// CHECK-NEXT:        ^8(%arg7 : index):
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
