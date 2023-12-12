// RUN: xdsl-opt %s -p "scf-parallel-loop-tiling{parallel-loop-tile-sizes=4,0,4}" --split-input-file | filecheck %s
// RUN: xdsl-opt %s -p "scf-parallel-loop-tiling{parallel-loop-tile-sizes=0,4,4}" --split-input-file | filecheck %s --check-prefix CHECK-FIRST
// COM: xdsl-opt %s -p "scf-parallel-loop-tiling{parallel-loop-tile-sizes=4,4,0}" --split-input-file | filecheck %s --check-prefix CHECK-LAST

func.func @tile_partial() {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %size = arith.constant 64 : index
    "scf.parallel"(%zero, %zero, %zero, %size, %size, %size, %one, %one, %one) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
    ^5(%arg2_2 : index, %arg3_2 : index, %arg4_2 : index):
    scf.yield
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
// CHECK-NEXT:   "scf.parallel"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:   ^{{.*}}(%{{.*}} : index, %{{.*}} : index):
// CHECK-NEXT:     %{{.*}} = "affine.min"(%{{.*}}, %{{.*}}, %{{.*}}) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-NEXT:     %{{.*}} = "affine.min"(%{{.*}}, %{{.*}}, %{{.*}}) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-NEXT:     "scf.parallel"(%{{.*}}, %zero, %{{.*}}, %{{.*}}, %size, %{{.*}}, %{{.*}}, %one, %{{.*}}) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:     ^{{.*}}(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index):
// CHECK-NEXT:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:     scf.yield
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
// CHECK-FIRST-NEXT:   "scf.parallel"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-FIRST-NEXT:   ^{{.*}}(%{{.*}} : index, %{{.*}} : index):
// CHECK-FIRST-NEXT:     %{{.*}} = "affine.min"(%{{.*}}, %{{.*}}, %{{.*}}) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-FIRST-NEXT:     %{{.*}} = "affine.min"(%{{.*}}, %{{.*}}, %{{.*}}) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-FIRST-NEXT:     "scf.parallel"(%zero, %{{.*}}, %{{.*}}, %size, %{{.*}}, %{{.*}}, %one, %{{.*}}, %{{.*}}) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// CHECK-FIRST-NEXT:     ^{{.*}}(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index):
// CHECK-FIRST-NEXT:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-FIRST-NEXT:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-FIRST-NEXT:       scf.yield
// CHECK-FIRST-NEXT:     }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-FIRST-NEXT:     scf.yield
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
// CHECK-LAST-NEXT:   "scf.parallel"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-LAST-NEXT:   ^{{.*}}(%{{.*}} : index, %{{.*}} : index):
// CHECK-LAST-NEXT:     %{{.*}} = "affine.min"(%{{.*}}, %{{.*}}, %{{.*}}) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-LAST-NEXT:     %{{.*}} = "affine.min"(%{{.*}}, %{{.*}}, %{{.*}}) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-LAST-NEXT:     "scf.parallel"(%{{.*}}, %{{.*}}, %zero, %{{.*}}, %{{.*}}, %size, %{{.*}}, %{{.*}}, %one) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// CHECK-LAST-NEXT:     ^{{.*}}(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index):
// CHECK-LAST-NEXT:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-LAST-NEXT:       %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-LAST-NEXT:       scf.yield
// CHECK-LAST-NEXT:     }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-LAST-NEXT:     scf.yield
// CHECK-LAST-NEXT:   }) : (index, index, index, index, index, index) -> ()
// CHECK-LAST-NEXT:   func.return
// CHECK-LAST-NEXT: }
