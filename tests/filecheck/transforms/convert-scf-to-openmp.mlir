// RUN: xdsl-opt -p convert-scf-to-openpm %s | filecheck %s

builtin.module {
  func.func @parallel(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index) {
    "scf.parallel"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
    ^0(%arg6 : index, %arg7 : index):
      "test.op"(%arg6, %arg7) : (index, index) -> ()
      scf.yield
    }) : (index, index, index, index, index, index) -> ()
    func.return
  }
  
// CHECK:         func.func @parallel(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index) {
// CHECK-NEXT:      "omp.parallel"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.wsloop"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        ^{{.*}}(%{{.*}} : index, %{{.*}} : index):
// CHECK-NEXT:          "memref.alloca_scope"() ({
// CHECK-NEXT:            "test.op"(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:            "memref.alloca_scope.return"() : () -> ()
// CHECK-NEXT:          }) : () -> ()
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @nested_loops(%arg0_1 : index, %arg1_1 : index, %arg2_1 : index, %arg3_1 : index, %arg4_1 : index, %arg5_1 : index) {
    "scf.parallel"(%arg0_1, %arg2_1, %arg4_1) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
    ^1(%arg6_1 : index):
      "scf.parallel"(%arg1_1, %arg3_1, %arg5_1) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
      ^2(%arg7_1 : index):
        "test.op"(%arg6_1, %arg7_1) : (index, index) -> ()
        scf.yield
      }) : (index, index, index) -> ()
      scf.yield
    }) : (index, index, index) -> ()
    func.return
  }

// CHECK:         func.func @nested_loops(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index) {
// CHECK-NEXT:      "omp.parallel"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.wsloop"(%{{.*}}, %{{.*}}, %{{.*}}) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        ^{{.*}}(%{{.*}} : index):
// CHECK-NEXT:          "memref.alloca_scope"() ({
// CHECK-NEXT:            "omp.parallel"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:              "omp.wsloop"(%{{.*}}, %{{.*}}, %{{.*}}) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>}> ({
// CHECK-NEXT:              ^{{.*}}(%{{.*}} : index):
// CHECK-NEXT:                "memref.alloca_scope"() ({
// CHECK-NEXT:                  "test.op"(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:                  "memref.alloca_scope.return"() : () -> ()
// CHECK-NEXT:                }) : () -> ()
// CHECK-NEXT:                omp.yield
// CHECK-NEXT:              }) : (index, index, index) -> ()
// CHECK-NEXT:              "omp.terminator"() : () -> ()
// CHECK-NEXT:            }) : () -> ()
// CHECK-NEXT:            "memref.alloca_scope.return"() : () -> ()
// CHECK-NEXT:          }) : () -> ()
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

  func.func @adjacent_loops(%arg0_2 : index, %arg1_2 : index, %arg2_2 : index, %arg3_2 : index, %arg4_2 : index, %arg5_2 : index) {
    "scf.parallel"(%arg0_2, %arg2_2, %arg4_2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
    ^3(%arg6_2 : index):
      "test.op"(%arg6_2) : (index) -> ()
      scf.yield
    }) : (index, index, index) -> ()
    "scf.parallel"(%arg1_2, %arg3_2, %arg5_2) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
    ^4(%arg6_3 : index):
      "test.op"(%arg6_3) : (index) -> ()
      scf.yield
    }) : (index, index, index) -> ()
    func.return
  }

// CHECK:         func.func @adjacent_loops(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index, %{{.*}} : index) {
// CHECK-NEXT:      "omp.parallel"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.wsloop"(%{{.*}}, %{{.*}}, %{{.*}}) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        ^{{.*}}(%{{.*}} : index):
// CHECK-NEXT:          "memref.alloca_scope"() ({
// CHECK-NEXT:            "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:            "memref.alloca_scope.return"() : () -> ()
// CHECK-NEXT:          }) : () -> ()
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "omp.parallel"() <{"operandSegmentSizes" = array<i32: 0, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        "omp.wsloop"(%{{.*}}, %{{.*}}, %{{.*}}) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0, 0, 0, 0>}> ({
// CHECK-NEXT:        ^{{.*}}(%{{.*}} : index):
// CHECK-NEXT:          "memref.alloca_scope"() ({
// CHECK-NEXT:            "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:            "memref.alloca_scope.return"() : () -> ()
// CHECK-NEXT:          }) : () -> ()
// CHECK-NEXT:          omp.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        "omp.terminator"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

}