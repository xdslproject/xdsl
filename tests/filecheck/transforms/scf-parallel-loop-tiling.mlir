// RUN: xdsl-opt %s -p "scf-parallel-loop-tiling{parallel-loop-tile-sizes=1,4}" --split-input-file | filecheck %s

func.func @parallel_loop(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : memref<?x?xf32>, %arg7 : memref<?x?xf32>, %arg8 : memref<?x?xf32>, %arg9 : memref<?x?xf32>) {
  "scf.parallel"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb0(%arg10 : index, %arg11 : index):
    %0 = memref.load %arg7[%arg10, %arg11] : memref<?x?xf32>
    %1 = memref.load %arg8[%arg10, %arg11] : memref<?x?xf32>
    %2 = arith.addf %0, %1 : f32
    memref.store %2, %arg9[%arg10, %arg11] : memref<?x?xf32>
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  func.return
}

// CHECK:         func @parallel_loop(
// CHECK-SAME:                        %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>) {
// CHECK:           %{{.*}} = arith.constant 0 : index
// CHECK:           %{{.*}} = arith.constant 1 : index
// CHECK:           %{{.*}} = arith.constant 4 : index
// CHECK:           %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:           %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:           "scf.parallel"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK:           ^{{.*}}({{%.*}} : index, {{%.*}} : index):
// CHECK:             %{{.*}} = "affine.min"(%{{.*}}, %{{.*}}, %{{.*}}) <{map = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK:             %{{.*}} = "affine.min"(%{{.*}}, %{{.*}}, %{{.*}}) <{map = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK:             "scf.parallel"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK:             ^{{.*}}({{%.*}} : index, {{%.*}} : index):
// CHECK:               %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK:               %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK:               %{{.*}} = memref.load %{{.*}}{{\[}}%{{.*}}, %{{.*}}] : memref<?x?xf32>
// CHECK:               %{{.*}} = memref.load %{{.*}}{{\[}}%{{.*}}, %{{.*}}] : memref<?x?xf32>
// CHECK:               %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK:               memref.store %{{.*}}, %{{.*}}{{\[}}%{{.*}}, %{{.*}}] : memref<?x?xf32>
// CHECK:             })
// CHECK:           })
// CHECK:           return

// -----

func.func @static_loop_with_step() {
  %3 = arith.constant 0 : index
  %4 = arith.constant 3 : index
  %5 = arith.constant 22 : index
  %6 = arith.constant 24 : index
  "scf.parallel"(%3, %3, %5, %6, %4, %4) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb1(%arg0_1 : index, %arg1_1 : index):
    "test.op"() : () -> ()
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  func.return
}

// CHECK-LABEL:   func @static_loop_with_step() {
// CHECK:           {{%.*}} = arith.constant 0 : index
// CHECK:           {{%.*}} = arith.constant 3 : index
// CHECK:           {{%.*}} = arith.constant 22 : index
// CHECK:           {{%.*}} = arith.constant 24 : index
// CHECK:           {{%.*}} = arith.constant 0 : index
// CHECK:           {{%.*}} = arith.constant 1 : index
// CHECK:           {{%.*}} = arith.constant 4 : index
// CHECK:           {{%.*}} = arith.muli {{%.*}}, {{%.*}} : index
// CHECK:           {{%.*}} = arith.muli {{%.*}}, {{%.*}} : index
// CHECK:           "scf.parallel"({{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK:           ^{{.*}}({{%.*}} : index, {{%.*}} : index):
// CHECK:             "scf.parallel"({{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK:             ^{{.*}}({{%.*}} : index, {{%.*}} : index):
// CHECK:               "test.op"() : () -> ()
// CHECK:               scf.reduce
// CHECK:             })
// CHECK:           })
// CHECK:           return

// -----

func.func @tile_nested_innermost() {
  %7 = arith.constant 2 : index
  %8 = arith.constant 0 : index
  %9 = arith.constant 1 : index
  "scf.parallel"(%8, %8, %7, %7, %9, %9) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb2(%arg0_2 : index, %arg1_2 : index):
    "scf.parallel"(%8, %8, %7, %7, %9, %9) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
    ^bb3(%arg2_1 : index, %arg3_1 : index):
      "test.op"() : () -> ()
      scf.reduce
    }) : (index, index, index, index, index, index) -> ()
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  "scf.parallel"(%8, %8, %7, %7, %9, %9) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
  ^bb4(%arg0_3 : index, %arg1_3 : index):
    "test.op"() : () -> ()
    scf.reduce
  }) : (index, index, index, index, index, index) -> ()
  func.return
}

// CHECK-LABEL:   func @tile_nested_innermost() {
// CHECK:           %{{.*}} = arith.constant 2 : index
// CHECK:           %{{.*}} = arith.constant 0 : index
// CHECK:           %{{.*}} = arith.constant 1 : index
// CHECK:           "scf.parallel"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK:             %{{.*}} = arith.constant 0 : index
// CHECK:             %{{.*}} = arith.constant 1 : index
// CHECK:             %{{.*}} = arith.constant 4 : index
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:             "scf.parallel"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK:               %{{.*}} = "affine.min"(%{{.*}}, %{{.*}}, %{{.*}}) <{map = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK:               "scf.parallel"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK:                 = arith.addi %{{.*}}, %{{.*}} : index
// CHECK:                 = arith.addi %{{.*}}, %{{.*}} : index
// CHEC:                  "test.op"() : () -> ()
// CHECK:               })
// CHECK:             })
// CHECK:           })
// CHECK:           %{{.*}} = arith.constant 0 : index
// CHECK:           %{{.*}} = arith.constant 1 : index
// CHECK:           %{{.*}} = arith.constant 4 : index
// CHECK:           %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:           %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK:           "scf.parallel"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK:             %{{.*}} = "affine.min"(%{{.*}}, %{{.*}}, %{{.*}}) <{map = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK:             "scf.parallel"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK:               = arith.addi %{{.*}}, %{{.*}} : index
// CHECK:               = arith.addi %{{.*}}, %{{.*}} : index
// CHECK:               "test.op"() : () -> ()
// CHECK:             })
// CHECK:           })
// CHECK:           return
// CHECK:         }

// -----

func.func @tile_nested_in_non_ploop() {
  %10 = arith.constant 0 : index
  %11 = arith.constant 1 : index
  %12 = arith.constant 2 : index
  scf.for %arg0_4 = %10 to %12 step %11 {
    scf.for %arg1_4 = %10 to %12 step %11 {
      "scf.parallel"(%10, %10, %12, %12, %11, %11) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
      ^bb5(%arg2_2 : index, %arg3_2 : index):
        "test.op"() : () -> ()
        scf.reduce
      }) : (index, index, index, index, index, index) -> ()
    }
  }
  func.return
}

// CHECK-LABEL: func @tile_nested_in_non_ploop
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             "scf.parallel"
// CHECK:               "scf.parallel"
// CHECK:                 "test.op"() : () -> ()
// CHECK:               })
// CHECK:             })
// CHECK:           }
// CHECK:         }
// CHECK:       }
