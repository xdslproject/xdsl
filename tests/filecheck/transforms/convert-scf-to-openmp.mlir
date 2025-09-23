// RUN: xdsl-opt -p convert-scf-to-openmp %s | filecheck %s
// RUN: xdsl-opt -p "convert-scf-to-openmp{nested=true}" %s | filecheck %s --check-prefix NESTED
// RUN: xdsl-opt -p "convert-scf-to-openmp{collapse=1}" %s | filecheck %s --check-prefix COLLAPSE
// RUN: xdsl-opt -p "convert-scf-to-openmp{schedule=dynamic}" %s | filecheck %s --check-prefix DYNAMIC
// RUN: xdsl-opt -p "convert-scf-to-openmp{chunk=4}" %s | filecheck %s --check-prefix CHUNK
// Check that a `collapse` greater than the loop depth doesn't crash
// RUN: xdsl-opt -p "convert-scf-to-openmp{collapse=3}" %s

builtin.module {
  func.func @parallel(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index) {
    "scf.parallel"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
    ^bb0(%arg6 : index, %arg7 : index):
      "test.op"(%arg6, %arg7) : (index, index) -> ()
      scf.reduce
    }) : (index, index, index, index, index, index) -> ()
    func.return
  }

// Check the default lowering.
// CHECK:         func.func @parallel(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index) {
// CHECK-NEXT:      "omp.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:          "omp.loop_nest"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ({
// CHECK-NEXT:          ^{{.*}}(%{{.*}} : index, %{{.*}} : index):
// CHECK-NEXT:            "memref.alloca_scope"() ({
// CHECK-NEXT:              "test.op"(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:              "memref.alloca_scope.return"() : () -> ()
// CHECK-NEXT:            }) : () -> ()
// CHECK-NEXT:            omp.yield
// CHECK-NEXT:          }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:        }) : () -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// Check that using `collapse=1` converts only the first dimension to OpenMP, and keeps the
// inner one(s) as an `scf.parallel` for any other further conversion.
// COLLAPSE:         func.func @parallel(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index) {
// COLLAPSE-NEXT:      "omp.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0>}> ({
// COLLAPSE-NEXT:        "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>}> ({
// COLLAPSE-NEXT:          "omp.loop_nest"(%{{.*}}, %{{.*}}, %{{.*}}) ({
// COLLAPSE-NEXT:          ^{{.*}}(%{{.*}} : index):
// COLLAPSE-NEXT:            "memref.alloca_scope"() ({
// COLLAPSE-NEXT:              "scf.parallel"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
// COLLAPSE-NEXT:              ^{{.*}}(%{{.*}} : index):
// COLLAPSE-NEXT:                "test.op"(%{{.*}}, %{{.*}}) : (index, index) -> ()
// COLLAPSE-NEXT:                scf.reduce
// COLLAPSE-NEXT:              }) : (index, index, index) -> ()
// COLLAPSE-NEXT:              "memref.alloca_scope.return"() : () -> ()
// COLLAPSE-NEXT:            }) : () -> ()
// COLLAPSE-NEXT:            omp.yield
// COLLAPSE-NEXT:          }) : (index, index, index) -> ()
// COLLAPSE-NEXT:        }) : () -> ()
// COLLAPSE-NEXT:        "omp.terminator"() : () -> ()
// COLLAPSE-NEXT:      }) : () -> ()
// COLLAPSE-NEXT:      func.return
// COLLAPSE-NEXT:    }

// Check that using `schedule` does set the OpenMP loop's schedule
// DYNAMIC:         func.func @parallel(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index) {
// DYNAMIC-NEXT:      "omp.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0>}> ({
// DYNAMIC-NEXT:        "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, schedule_kind = #omp<schedulekind dynamic>}> ({
// DYNAMIC-NEXT:          "omp.loop_nest"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ({
// DYNAMIC-NEXT:          ^{{.*}}(%{{.*}} : index, %{{.*}} : index):
// DYNAMIC-NEXT:            "memref.alloca_scope"() ({
// DYNAMIC-NEXT:              "test.op"(%{{.*}}, %{{.*}}) : (index, index) -> ()
// DYNAMIC-NEXT:              "memref.alloca_scope.return"() : () -> ()
// DYNAMIC-NEXT:            }) : () -> ()
// DYNAMIC-NEXT:            omp.yield
// DYNAMIC-NEXT:          }) : (index, index, index, index, index, index) -> ()
// DYNAMIC-NEXT:        }) : () -> ()
// DYNAMIC-NEXT:        "omp.terminator"() : () -> ()
// DYNAMIC-NEXT:      }) : () -> ()
// DYNAMIC-NEXT:      func.return
// DYNAMIC-NEXT:    }

// Check that using `chunk` is setting the OpenMP loop's chunk size.
// Also, check that doing so without selecting a scheule sets it to static.
// (It is invalid to set a chunk size without setting a schedule)
// CHUNK:         func.func @parallel(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index) {
// CHUNK-NEXT:      "omp.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0>}> ({
// CHUNK-NEXT:        %{{.*}} = arith.constant 4 : index
// CHUNK-NEXT:        "omp.wsloop"(%{{.*}}) <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 1>, schedule_kind = #omp<schedulekind static>}> ({
// CHUNK-NEXT:          "omp.loop_nest"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) ({
// CHUNK-NEXT:          ^{{.*}}(%{{.*}} : index, %{{.*}} : index):
// CHUNK-NEXT:            "memref.alloca_scope"() ({
// CHUNK-NEXT:              "test.op"(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHUNK-NEXT:              "memref.alloca_scope.return"() : () -> ()
// CHUNK-NEXT:            }) : () -> ()
// CHUNK-NEXT:            omp.yield
// CHUNK-NEXT:          }) : (index, index, index, index, index, index) -> ()
// CHUNK-NEXT:        }) : (index) -> ()
// CHUNK-NEXT:        "omp.terminator"() : () -> ()
// CHUNK-NEXT:      }) : () -> ()
// CHUNK-NEXT:      func.return
// CHUNK-NEXT:    }

  func.func @nested_loops(%arg0_1 : index, %arg1_1 : index, %arg2_1 : index, %arg3_1 : index, %arg4_1 : index, %arg5_1 : index) {
    "scf.parallel"(%arg0_1, %arg2_1, %arg4_1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb1(%arg6_1 : index):
      "scf.parallel"(%arg1_1, %arg3_1, %arg5_1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
      ^bb2(%arg7_1 : index):
        "test.op"(%arg6_1, %arg7_1) : (index, index) -> ()
        scf.reduce
      }) : (index, index, index) -> ()
      scf.reduce
    }) : (index, index, index) -> ()
    func.return
  }

// Check that the default conversion does not convert the nested loop.
// CHECK-NEXT:    func.func @nested_loops(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index) {
// CHECK-NEXT:      "omp.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:          "omp.loop_nest"(%{{.*}}, %{{.*}}, %{{.*}}) ({
// CHECK-NEXT:          ^{{.*}}(%{{.*}} : index):
// CHECK-NEXT:            "memref.alloca_scope"() ({
// CHECK-NEXT:              "scf.parallel"(%{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:              ^{{.*}}(%{{.*}} : index):
// CHECK-NEXT:                "test.op"(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:                scf.reduce
// CHECK-NEXT:              }) : (index, index, index) -> ()
// CHECK-NEXT:              "memref.alloca_scope.return"() : () -> ()
// CHECK-NEXT:            }) : () -> ()
// CHECK-NEXT:            omp.yield
// CHECK-NEXT:          }) : (index, index, index) -> ()
// CHECK-NEXT:        }) : () -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// Check that using `nested=true` allows to lower the nested loop.
// NESTED:         func.func @nested_loops(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index) {
// NESTED-NEXT:      "omp.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0>}> ({
// NESTED-NEXT:        "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>}> ({
// NESTED-NEXT:          "omp.loop_nest"(%{{.*}}, %{{.*}}, %{{.*}}) ({
// NESTED-NEXT:          ^{{.*}}(%{{.*}} : index):
// NESTED-NEXT:            "memref.alloca_scope"() ({
// NESTED-NEXT:              "omp.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0>}> ({
// NESTED-NEXT:                "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>}> ({
// NESTED-NEXT:                  "omp.loop_nest"(%{{.*}}, %{{.*}}, %{{.*}}) ({
// NESTED-NEXT:                  ^{{.*}}(%{{.*}} : index):
// NESTED-NEXT:                    "memref.alloca_scope"() ({
// NESTED-NEXT:                      "test.op"(%{{.*}}, %{{.*}}) : (index, index) -> ()
// NESTED-NEXT:                      "memref.alloca_scope.return"() : () -> ()
// NESTED-NEXT:                    }) : () -> ()
// NESTED-NEXT:                    omp.yield
// NESTED-NEXT:                  }) : (index, index, index) -> ()
// NESTED-NEXT:                }) : () -> ()
// NESTED-NEXT:                "omp.terminator"() : () -> ()
// NESTED-NEXT:              }) : () -> ()
// NESTED-NEXT:              "memref.alloca_scope.return"() : () -> ()
// NESTED-NEXT:            }) : () -> ()
// NESTED-NEXT:            omp.yield
// NESTED-NEXT:          }) : (index, index, index) -> ()
// NESTED-NEXT:        }) : () -> ()
// NESTED-NEXT:        "omp.terminator"() : () -> ()
// NESTED-NEXT:      }) : () -> ()
// NESTED-NEXT:      func.return
// NESTED-NEXT:    }

  func.func @adjacent_loops(%arg0_2 : index, %arg1_2 : index, %arg2_2 : index, %arg3_2 : index, %arg4_2 : index, %arg5_2 : index) {
    "scf.parallel"(%arg0_2, %arg2_2, %arg4_2) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb3(%arg6_2 : index):
      "test.op"(%arg6_2) : (index) -> ()
      scf.reduce
    }) : (index, index, index) -> ()
    "scf.parallel"(%arg1_2, %arg3_2, %arg5_2) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb4(%arg6_3 : index):
      "test.op"(%arg6_3) : (index) -> ()
      scf.reduce
    }) : (index, index, index) -> ()
    func.return
  }

// Just another example, copied from MLIR's filecheck.
// CHECK:         func.func @adjacent_loops(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index) {
// CHECK-NEXT:      "omp.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:          "omp.loop_nest"(%{{.*}}, %{{.*}}, %{{.*}}) ({
// CHECK-NEXT:          ^{{.*}}(%{{.*}} : index):
// CHECK-NEXT:            "memref.alloca_scope"() ({
// CHECK-NEXT:              "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:              "memref.alloca_scope.return"() : () -> ()
// CHECK-NEXT:            }) : () -> ()
// CHECK-NEXT:            omp.yield
// CHECK-NEXT:          }) : (index, index, index) -> ()
// CHECK-NEXT:        }) : () -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:          "omp.loop_nest"(%{{.*}}, %{{.*}}, %{{.*}}) ({
// CHECK-NEXT:          ^{{.*}}(%{{.*}} : index):
// CHECK-NEXT:            "memref.alloca_scope"() ({
// CHECK-NEXT:              "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:              "memref.alloca_scope.return"() : () -> ()
// CHECK-NEXT:            }) : () -> ()
// CHECK-NEXT:            omp.yield
// CHECK-NEXT:          }) : (index, index, index) -> ()
// CHECK-NEXT:        }) : () -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @reduction1(%arg0_1 : index, %arg1_1 : index, %arg2_1 : index, %arg3_1 : index, %arg4_1 : index) {
    %0 = arith.constant 1 : index
    %1 = arith.constant 0.000000e+00 : f32
    %2 = "scf.parallel"(%arg0_1, %arg1_1, %arg2_1, %arg3_1, %arg4_1, %0, %1) <{operandSegmentSizes = array<i32: 2, 2, 2, 1>}> ({
    ^bb5(%arg5_1 : index, %arg6_1 : index):
      %3 = arith.constant 1.000000e+00 : f32
      scf.reduce(%3 : f32) {
      ^bb6(%arg7_1 : f32, %arg8 : f32):
        %4 = arith.addf %arg7_1, %arg8 : f32
        scf.reduce.return %4 : f32
      }
    }) : (index, index, index, index, index, index, f32) -> f32
    func.return
  }

// Check that the pass doesn't crash on reductions, but just safely ignores them for now.
// CHECK:         func.func @reduction1(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index) {
// CHECK-NEXT:      %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:      %{{.*}} = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %{{.*}} = "scf.parallel"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 2, 2, 1>}> ({
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : index, %{{.*}} : index):
// CHECK-NEXT:        %{{.*}} = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:        scf.reduce(%{{.*}} : f32) {
// CHECK-NEXT:        ^{{.*}}(%{{.*}} : f32, %{{.*}} : f32):
// CHECK-NEXT:          %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:          scf.reduce.return %{{.*}} : f32
// CHECK-NEXT:        }
// CHECK-NEXT:      }) : (index, index, index, index, index, index, f32) -> f32
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
}
