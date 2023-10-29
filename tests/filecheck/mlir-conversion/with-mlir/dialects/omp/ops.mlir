// RUN: xdsl-opt %s --print-op-generic | mlir-opt --mlir-print-op-generic | xdsl-opt | filecheck %s

builtin.module {
  func.func @omp_ordered(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i64, %arg4 : i64, %arg5 : i64, %arg6 : i64) {
    "omp.wsloop"(%arg0, %arg1, %arg2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>, "ordered_val" = 0 : i64}> ({
    ^0(%arg7 : i32):
      omp.yield
    }) : (i32, i32, i32) -> ()
    "omp.wsloop"(%arg0, %arg1, %arg2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>, "ordered_val" = 1 : i64}> ({
    ^1(%arg7_1 : i32):
      omp.yield
    }) : (i32, i32, i32) -> ()
    "omp.wsloop"(%arg0, %arg1, %arg2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>, "ordered_val" = 2 : i64}> ({
    ^2(%arg7_2 : i32):
      omp.yield
    }) : (i32, i32, i32) -> ()
    func.return
  }
  func.func @omp_wsloop(%arg0_1 : index, %arg1_1 : index, %arg2_1 : index, %arg3_1 : memref<1xi32>, %arg4_1 : i32, %arg5_1 : i32) {
    "omp.wsloop"(%arg0_1, %arg1_1, %arg2_1) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>, "ordered_val" = 1 : i64}> ({
    ^3(%arg6_1 : index):
      omp.yield
    }) : (index, index, index) -> ()
    "omp.wsloop"(%arg0_1, %arg1_1, %arg2_1, %arg3_1, %arg4_1) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 1, 1, 0, 0>, "schedule_val" = #omp<schedulekind static>}> ({
    ^4(%arg6_2 : index):
      omp.yield
    }) : (index, index, index, memref<1xi32>, i32) -> ()
    "omp.wsloop"(%arg0_1, %arg1_1, %arg2_1, %arg3_1, %arg3_1, %arg4_1, %arg4_1) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 2, 2, 0, 0>, "schedule_val" = #omp<schedulekind static>}> ({
    ^5(%arg6_3 : index):
      omp.yield
    }) : (index, index, index, memref<1xi32>, memref<1xi32>, i32, i32) -> ()
    "omp.wsloop"(%arg0_1, %arg1_1, %arg2_1, %arg3_1, %arg4_1, %arg5_1) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 1, 1, 0, 1>, "ordered_val" = 2 : i64, "schedule_val" = #omp<schedulekind dynamic>}> ({
    ^6(%arg6_4 : index):
      omp.yield
    }) : (index, index, index, memref<1xi32>, i32, i32) -> ()
    "omp.wsloop"(%arg0_1, %arg1_1, %arg2_1) <{"nowait", "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>, "schedule_val" = #omp<schedulekind auto>}> ({
    ^7(%arg6_5 : index):
      omp.yield
    }) : (index, index, index) -> ()
    func.return
  }
  func.func @omp_wsloop_pretty(%arg0_2 : index, %arg1_2 : index, %arg2_2 : index, %arg3_2 : memref<1xi32>, %arg4_2 : i32, %arg5_2 : i32, %arg6_6 : i16) {
    "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>, "ordered_val" = 2 : i64}> ({
    ^8(%arg7_3 : index):
      omp.yield
    }) : (index, index, index) -> ()
    "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2, %arg3_2, %arg4_2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 1, 1, 0, 0>, "schedule_val" = #omp<schedulekind static>}> ({
    ^9(%arg7_4 : index):
      omp.yield
    }) : (index, index, index, memref<1xi32>, i32) -> ()
    "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2, %arg3_2, %arg4_2, %arg5_2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 1, 1, 0, 1>, "ordered_val" = 2 : i64, "schedule_val" = #omp<schedulekind static>}> ({
    ^10(%arg7_5 : index):
      omp.yield
    }) : (index, index, index, memref<1xi32>, i32, i32) -> ()
    "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2, %arg3_2, %arg4_2, %arg5_2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 1, 1, 0, 1>, "ordered_val" = 2 : i64, "schedule_modifier" = #omp<sched_mod nonmonotonic>, "schedule_val" = #omp<schedulekind dynamic>}> ({
    ^11(%arg7_6 : index):
      omp.yield
    }) : (index, index, index, memref<1xi32>, i32, i32) -> ()
    "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2, %arg3_2, %arg4_2, %arg6_6) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 1, 1, 0, 1>, "ordered_val" = 2 : i64, "schedule_modifier" = #omp<sched_mod monotonic>, "schedule_val" = #omp<schedulekind dynamic>}> ({
    ^12(%arg7_7 : index):
      omp.yield
    }) : (index, index, index, memref<1xi32>, i32, i16) -> ()
    "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>}> ({
    ^13(%arg7_8 : index):
      omp.yield
    }) : (index, index, index) -> ()
    "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2) <{"inclusive", "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>}> ({
    ^14(%arg7_9 : index):
      omp.yield
    }) : (index, index, index) -> ()
    "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2) <{"nowait", "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>}> ({
    ^15(%arg7_10 : index):
      omp.yield
    }) : (index, index, index) -> ()
    "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2) <{"nowait", "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>, "order_val" = #omp<orderkind concurrent>}> ({
    ^16(%arg7_11 : index):
      omp.yield
    }) : (index, index, index) -> ()
    func.return
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @omp_ordered(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i64, %arg4 : i64, %arg5 : i64, %arg6 : i64) {
// CHECK-NEXT:      "omp.wsloop"(%arg0, %arg1, %arg2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>, "ordered_val" = 0 : i64}> ({
// CHECK-NEXT:      ^0(%arg7 : i32):
// CHECK-NEXT:        omp.yield
// CHECK-NEXT:      }) : (i32, i32, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg0, %arg1, %arg2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>, "ordered_val" = 1 : i64}> ({
// CHECK-NEXT:      ^1(%arg7_1 : i32):
// CHECK-NEXT:        omp.yield
// CHECK-NEXT:      }) : (i32, i32, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg0, %arg1, %arg2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>, "ordered_val" = 2 : i64}> ({
// CHECK-NEXT:      ^2(%arg7_2 : i32):
// CHECK-NEXT:        omp.yield
// CHECK-NEXT:      }) : (i32, i32, i32) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @omp_wsloop(%arg0_1 : index, %arg1_1 : index, %arg2_1 : index, %arg3_1 : memref<1xi32>, %arg4_1 : i32, %arg5_1 : i32) {
// CHECK-NEXT:      "omp.wsloop"(%arg0_1, %arg1_1, %arg2_1) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>, "ordered_val" = 1 : i64}> ({
// CHECK-NEXT:      ^3(%arg6_1 : index):
// CHECK-NEXT:        omp.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg0_1, %arg1_1, %arg2_1, %arg3_1, %arg4_1) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 1, 1, 0, 0>, "schedule_val" = #omp<schedulekind static>}> ({
// CHECK-NEXT:      ^4(%arg6_2 : index):
// CHECK-NEXT:        omp.yield
// CHECK-NEXT:      }) : (index, index, index, memref<1xi32>, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg0_1, %arg1_1, %arg2_1, %arg3_1, %arg3_1, %arg4_1, %arg4_1) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 2, 2, 0, 0>, "schedule_val" = #omp<schedulekind static>}> ({
// CHECK-NEXT:      ^5(%arg6_3 : index):
// CHECK-NEXT:        omp.yield
// CHECK-NEXT:      }) : (index, index, index, memref<1xi32>, memref<1xi32>, i32, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg0_1, %arg1_1, %arg2_1, %arg3_1, %arg4_1, %arg5_1) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 1, 1, 0, 1>, "ordered_val" = 2 : i64, "schedule_val" = #omp<schedulekind dynamic>}> ({
// CHECK-NEXT:      ^6(%arg6_4 : index):
// CHECK-NEXT:        omp.yield
// CHECK-NEXT:      }) : (index, index, index, memref<1xi32>, i32, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg0_1, %arg1_1, %arg2_1) <{"nowait", "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>, "schedule_val" = #omp<schedulekind auto>}> ({
// CHECK-NEXT:      ^7(%arg6_5 : index):
// CHECK-NEXT:        omp.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @omp_wsloop_pretty(%arg0_2 : index, %arg1_2 : index, %arg2_2 : index, %arg3_2 : memref<1xi32>, %arg4_2 : i32, %arg5_2 : i32, %arg6_6 : i16) {
// CHECK-NEXT:      "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>, "ordered_val" = 2 : i64}> ({
// CHECK-NEXT:      ^8(%arg7_3 : index):
// CHECK-NEXT:        omp.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2, %arg3_2, %arg4_2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 1, 1, 0, 0>, "schedule_val" = #omp<schedulekind static>}> ({
// CHECK-NEXT:      ^9(%arg7_4 : index):
// CHECK-NEXT:        omp.yield
// CHECK-NEXT:      }) : (index, index, index, memref<1xi32>, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2, %arg3_2, %arg4_2, %arg5_2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 1, 1, 0, 1>, "ordered_val" = 2 : i64, "schedule_val" = #omp<schedulekind static>}> ({
// CHECK-NEXT:      ^10(%arg7_5 : index):
// CHECK-NEXT:        omp.yield
// CHECK-NEXT:      }) : (index, index, index, memref<1xi32>, i32, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2, %arg3_2, %arg4_2, %arg5_2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 1, 1, 0, 1>, "ordered_val" = 2 : i64, "schedule_modifier" = #omp<sched_mod nonmonotonic>, "schedule_val" = #omp<schedulekind dynamic>}> ({
// CHECK-NEXT:      ^11(%arg7_6 : index):
// CHECK-NEXT:        omp.yield
// CHECK-NEXT:      }) : (index, index, index, memref<1xi32>, i32, i32) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2, %arg3_2, %arg4_2, %arg6_6) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 1, 1, 0, 1>, "ordered_val" = 2 : i64, "schedule_modifier" = #omp<sched_mod monotonic>, "schedule_val" = #omp<schedulekind dynamic>}> ({
// CHECK-NEXT:      ^12(%arg7_7 : index):
// CHECK-NEXT:        omp.yield
// CHECK-NEXT:      }) : (index, index, index, memref<1xi32>, i32, i16) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>}> ({
// CHECK-NEXT:      ^13(%arg7_8 : index):
// CHECK-NEXT:        omp.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2) <{"inclusive", "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>}> ({
// CHECK-NEXT:      ^14(%arg7_9 : index):
// CHECK-NEXT:        omp.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2) <{"nowait", "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>}> ({
// CHECK-NEXT:      ^15(%arg7_10 : index):
// CHECK-NEXT:        omp.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      "omp.wsloop"(%arg0_2, %arg1_2, %arg2_2) <{"nowait", "operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>, "order_val" = #omp<orderkind concurrent>}> ({
// CHECK-NEXT:      ^16(%arg7_11 : index):
// CHECK-NEXT:        omp.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
