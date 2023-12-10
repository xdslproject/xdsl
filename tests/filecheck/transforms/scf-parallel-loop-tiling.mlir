// RUN: xdsl-opt %s -p "scf-parallel-loop-tiling{parallel-loop-tile-sizes=1,4}" --split-input-file | filecheck %s

func.func @static_loop_with_step() {
  %3 = arith.constant 0 : index
  %4 = arith.constant 3 : index
  %5 = arith.constant 22 : index
  %6 = arith.constant 24 : index
  "scf.parallel"(%3, %3, %5, %6, %4, %4) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
  ^1(%arg0_1 : index, %arg1_1 : index):
    scf.yield
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
// CHECK:           "scf.parallel"({{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK:           ^{{.*}}({{%.*}} : index, {{%.*}} : index):
// CHECK:             "scf.parallel"({{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK:             ^{{.*}}({{%.*}} : index, {{%.*}} : index):
// CHECK:               scf.yield
// CHECK:             })
// CHECK:           })
// CHECK:           return

// -----

func.func @tile_nested_in_non_ploop() {
  %10 = arith.constant 0 : index
  %11 = arith.constant 1 : index
  %12 = arith.constant 2 : index
  scf.for %arg0_4 = %10 to %12 step %11 {
    scf.for %arg1_4 = %10 to %12 step %11 {
      "scf.parallel"(%10, %10, %12, %12, %11, %11) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
      ^5(%arg2_2 : index, %arg3_2 : index):
        scf.yield
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
// CHECK:               })
// CHECK:             })
// CHECK:           }
// CHECK:         }
// CHECK:       }
