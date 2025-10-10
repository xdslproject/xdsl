// RUN: xdsl-opt %s -p "scf-parallel-loop-tiling{parallel-loop-tile-sizes=4,0,4}" | filecheck %s
// RUN: xdsl-opt %s -p "scf-parallel-loop-tiling{parallel-loop-tile-sizes=0,4,4}" | filecheck %s --check-prefix CHECK-FIRST
// RUN: xdsl-opt %s -p "scf-parallel-loop-tiling{parallel-loop-tile-sizes=4,4,0}" | filecheck %s --check-prefix CHECK-LAST

func.func @tile_partial() {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %size = arith.constant 64 : index
    "scf.parallel"(%zero, %zero, %zero, %size, %size, %size, %one, %one, %one) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
    ^bb5(%arg2_2 : index, %arg3_2 : index, %arg4_2 : index):
    scf.reduce
    }) : (index, index, index, index, index, index, index, index, index) -> ()
  func.return
}

// CHECK:      func.func @tile_partial() {
// CHECK-NEXT:   %zero = arith.constant 0 : index
// CHECK-NEXT:   %one = arith.constant 1 : index
// CHECK-NEXT:   %size = arith.constant 64 : index
// CHECK-NEXT:   %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:   %{{.*}} = arith.constant 4 : index
// CHECK-NEXT:   %{{.*}} = arith.constant 4 : index
// CHECK-NEXT:   %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   "scf.parallel"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:   ^{{.*}}(%{{.*}} : index, %{{.*}} : index):
// CHECK-NEXT:     %{{.*}} = "affine.min"(%{{.*}}, %{{.*}}, %{{.*}}) <{map = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-NEXT:     %{{.*}} = "affine.min"(%{{.*}}, %{{.*}}, %{{.*}}) <{map = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-NEXT:     "scf.parallel"(%{{.*}}, %zero, %{{.*}}, %{{.*}}, %size, %{{.*}}, %{{.*}}, %one, %{{.*}}) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:     ^{{.*}}(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index):
// CHECK-NEXT:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       scf.reduce
// CHECK-NEXT:     }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:     scf.reduce
// CHECK-NEXT:   }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:   func.return
// CHECK-NEXT: }

// CHECK-FIRST:      func.func @tile_partial() {
// CHECK-FIRST-NEXT:   %zero = arith.constant 0 : index
// CHECK-FIRST-NEXT:   %one = arith.constant 1 : index
// CHECK-FIRST-NEXT:   %size = arith.constant 64 : index
// CHECK-FIRST-NEXT:   %{{.*}} = arith.constant 0 : index
// CHECK-FIRST-NEXT:   %{{.*}} = arith.constant 4 : index
// CHECK-FIRST-NEXT:   %{{.*}} = arith.constant 4 : index
// CHECK-FIRST-NEXT:   %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK-FIRST-NEXT:   %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK-FIRST-NEXT:   "scf.parallel"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK-FIRST-NEXT:   ^{{.*}}(%{{.*}} : index, %{{.*}} : index):
// CHECK-FIRST-NEXT:     %{{.*}} = "affine.min"(%{{.*}}, %{{.*}}, %{{.*}}) <{map = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-FIRST-NEXT:     %{{.*}} = "affine.min"(%{{.*}}, %{{.*}}, %{{.*}}) <{map = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-FIRST-NEXT:     "scf.parallel"(%zero, %{{.*}}, %{{.*}}, %size, %{{.*}}, %{{.*}}, %one, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
// CHECK-FIRST-NEXT:     ^{{.*}}(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index):
// CHECK-FIRST-NEXT:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-FIRST-NEXT:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-FIRST-NEXT:       scf.reduce
// CHECK-FIRST-NEXT:     }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-FIRST-NEXT:     scf.reduce
// CHECK-FIRST-NEXT:   }) : (index, index, index, index, index, index) -> ()
// CHECK-FIRST-NEXT:   func.return
// CHECK-FIRST-NEXT: }

// CHECK-LAST:      func.func @tile_partial() {
// CHECK-LAST-NEXT:   %zero = arith.constant 0 : index
// CHECK-LAST-NEXT:   %one = arith.constant 1 : index
// CHECK-LAST-NEXT:   %size = arith.constant 64 : index
// CHECK-LAST-NEXT:   %{{.*}} = arith.constant 0 : index
// CHECK-LAST-NEXT:   %{{.*}} = arith.constant 4 : index
// CHECK-LAST-NEXT:   %{{.*}} = arith.constant 4 : index
// CHECK-LAST-NEXT:   %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK-LAST-NEXT:   %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK-LAST-NEXT:   "scf.parallel"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 2, 2, 2, 0>}> ({
// CHECK-LAST-NEXT:   ^{{.*}}(%{{.*}} : index, %{{.*}} : index):
// CHECK-LAST-NEXT:     %{{.*}} = "affine.min"(%{{.*}}, %{{.*}}, %{{.*}}) <{map = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-LAST-NEXT:     %{{.*}} = "affine.min"(%{{.*}}, %{{.*}}, %{{.*}}) <{map = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-LAST-NEXT:     "scf.parallel"(%{{.*}}, %{{.*}}, %zero, %{{.*}}, %{{.*}}, %size, %{{.*}}, %{{.*}}, %one) <{operandSegmentSizes = array<i32: 3, 3, 3, 0>}> ({
// CHECK-LAST-NEXT:     ^{{.*}}(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index):
// CHECK-LAST-NEXT:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-LAST-NEXT:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-LAST-NEXT:       scf.reduce
// CHECK-LAST-NEXT:     }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-LAST-NEXT:     scf.reduce
// CHECK-LAST-NEXT:   }) : (index, index, index, index, index, index) -> ()
// CHECK-LAST-NEXT:   func.return
// CHECK-LAST-NEXT: }

func.func @tile_partial_1d() {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %size = arith.constant 64 : index
    "scf.parallel"(%zero, %size, %one) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
    ^bb5(%arg1: index):
    scf.reduce
    }) : (index, index, index) -> ()
  func.return
}

// CHECK:         func.func @tile_partial_1d() {
// CHECK-NEXT:      %zero = arith.constant 0 : index
// CHECK-NEXT:      %one = arith.constant 1 : index
// CHECK-NEXT:      %size = arith.constant 64 : index
// CHECK-NEXT:      %0 = arith.constant 0 : index
// CHECK-NEXT:      %1 = arith.constant 4 : index
// CHECK-NEXT:      %2 = arith.muli %one, %1 : index
// CHECK-NEXT:      "scf.parallel"(%zero, %size, %2) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^bb0(%3 : index):
// CHECK-NEXT:        %4 = "affine.min"(%1, %size, %3) <{map = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-NEXT:        "scf.parallel"(%0, %4, %one) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:        ^bb1(%arg1 : index):
// CHECK-NEXT:          %5 = arith.addi %3, %arg1 : index
// CHECK-NEXT:          scf.reduce
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        scf.reduce
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK-FIRST:         func.func @tile_partial_1d() {
// CHECK-FIRST-NEXT:      %zero = arith.constant 0 : index
// CHECK-FIRST-NEXT:      %one = arith.constant 1 : index
// CHECK-FIRST-NEXT:      %size = arith.constant 64 : index
// CHECK-FIRST-NEXT:      "scf.parallel"(%zero, %size, %one) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
// CHECK-FIRST-NEXT:      ^bb0(%arg1 : index):
// CHECK-FIRST-NEXT:        scf.reduce
// CHECK-FIRST-NEXT:      }) : (index, index, index) -> ()
// CHECK-FIRST-NEXT:      func.return
// CHECK-FIRST-NEXT:    }
// CHECK-FIRST-NEXT:  }

// CHECK-LAST:         func.func @tile_partial_1d() {
// CHECK-LAST-NEXT:      %zero = arith.constant 0 : index
// CHECK-LAST-NEXT:      %one = arith.constant 1 : index
// CHECK-LAST-NEXT:      %size = arith.constant 64 : index
// CHECK-LAST-NEXT:      %0 = arith.constant 0 : index
// CHECK-LAST-NEXT:      %1 = arith.constant 4 : index
// CHECK-LAST-NEXT:      %2 = arith.muli %one, %1 : index
// CHECK-LAST-NEXT:      "scf.parallel"(%zero, %size, %2) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
// CHECK-LAST-NEXT:      ^bb0(%3 : index):
// CHECK-LAST-NEXT:        %4 = "affine.min"(%1, %size, %3) <{map = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-LAST-NEXT:        "scf.parallel"(%0, %4, %one) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
// CHECK-LAST-NEXT:        ^bb1(%arg1 : index):
// CHECK-LAST-NEXT:          %5 = arith.addi %3, %arg1 : index
// CHECK-LAST-NEXT:          scf.reduce
// CHECK-LAST-NEXT:        }) : (index, index, index) -> ()
// CHECK-LAST-NEXT:        scf.reduce
// CHECK-LAST-NEXT:      }) : (index, index, index) -> ()
// CHECK-LAST-NEXT:      func.return
// CHECK-LAST-NEXT:    }
// CHECK-LAST-NEXT:  }
