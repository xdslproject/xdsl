// RUN: python -m toy %s --emit=affine --accelerate --ir | filecheck %s

builtin.module {
  func.func @main() {
    %0 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<3x2xf64>
    %1 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<3x2xf64>
    %2 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<3x2xf64>
    %3 = arith.constant 1.000000e+00 : f64
    %4 = arith.constant 2.000000e+00 : f64
    %5 = arith.constant 3.000000e+00 : f64
    %6 = arith.constant 4.000000e+00 : f64
    %7 = arith.constant 5.000000e+00 : f64
    %8 = arith.constant 6.000000e+00 : f64
    %9 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<2x3xf64>
    %10 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<2x3xf64>
    "affine.store"(%3, %10) {"map" = affine_map<() -> (0, 0)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%4, %10) {"map" = affine_map<() -> (0, 1)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%5, %10) {"map" = affine_map<() -> (0, 2)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%6, %10) {"map" = affine_map<() -> (1, 0)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%7, %10) {"map" = affine_map<() -> (1, 1)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%8, %10) {"map" = affine_map<() -> (1, 2)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%3, %9) {"map" = affine_map<() -> (0, 0)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%4, %9) {"map" = affine_map<() -> (0, 1)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%5, %9) {"map" = affine_map<() -> (0, 2)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%6, %9) {"map" = affine_map<() -> (1, 0)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%7, %9) {"map" = affine_map<() -> (1, 1)>} : (f64, memref<2x3xf64>) -> ()
    "affine.store"(%8, %9) {"map" = affine_map<() -> (1, 2)>} : (f64, memref<2x3xf64>) -> ()
    "affine.for"() ({
    ^0(%arg0 : index):
      "affine.for"() ({
      ^1(%arg1 : index):
        %11 = "affine.load"(%9, %arg1, %arg0) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (memref<2x3xf64>, index, index) -> f64
        "affine.store"(%11, %2, %arg0, %arg1) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (f64, memref<3x2xf64>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {"lower_bound" = affine_map<() -> (0)>, "step" = 1 : index, "upper_bound" = affine_map<() -> (2)>} : () -> ()
      "affine.yield"() : () -> ()
    }) {"lower_bound" = affine_map<() -> (0)>, "step" = 1 : index, "upper_bound" = affine_map<() -> (3)>} : () -> ()
    "affine.for"() ({
    ^2(%arg0_1 : index):
      "affine.for"() ({
      ^3(%arg1_1 : index):
        %12 = "affine.load"(%10, %arg1_1, %arg0_1) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (memref<2x3xf64>, index, index) -> f64
        "affine.store"(%12, %1, %arg0_1, %arg1_1) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (f64, memref<3x2xf64>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {"lower_bound" = affine_map<() -> (0)>, "step" = 1 : index, "upper_bound" = affine_map<() -> (2)>} : () -> ()
      "affine.yield"() : () -> ()
    }) {"lower_bound" = affine_map<() -> (0)>, "step" = 1 : index, "upper_bound" = affine_map<() -> (3)>} : () -> ()
    "affine.for"() ({
    ^4(%arg0_2 : index):
      "affine.for"() ({
      ^5(%arg1_2 : index):
        %13 = "affine.load"(%2, %arg0_2, %arg1_2) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (memref<3x2xf64>, index, index) -> f64
        %14 = "affine.load"(%1, %arg0_2, %arg1_2) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (memref<3x2xf64>, index, index) -> f64
        %15 = arith.mulf %13, %14 : f64
        "affine.store"(%15, %0, %arg0_2, %arg1_2) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (f64, memref<3x2xf64>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {"lower_bound" = affine_map<() -> (0)>, "step" = 1 : index, "upper_bound" = affine_map<() -> (2)>} : () -> ()
      "affine.yield"() : () -> ()
    }) {"lower_bound" = affine_map<() -> (0)>, "step" = 1 : index, "upper_bound" = affine_map<() -> (3)>} : () -> ()
    printf.print_format "{}", %0 : memref<3x2xf64>
    "memref.dealloc"(%10) : (memref<2x3xf64>) -> ()
    "memref.dealloc"(%9) : (memref<2x3xf64>) -> ()
    "memref.dealloc"(%2) : (memref<3x2xf64>) -> ()
    "memref.dealloc"(%1) : (memref<3x2xf64>) -> ()
    "memref.dealloc"(%0) : (memref<3x2xf64>) -> ()
    func.return
  }
}

// CHECK:         builtin.module {
// CHECK-NEXT:      func.func @main() {
// CHECK-NEXT:        %0 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<3x2xf64>
// CHECK-NEXT:        %1 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<3x2xf64>
// CHECK-NEXT:        %2 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<3x2xf64>
// CHECK-NEXT:        %3 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:        %4 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:        %5 = arith.constant 3.000000e+00 : f64
// CHECK-NEXT:        %6 = arith.constant 4.000000e+00 : f64
// CHECK-NEXT:        %7 = arith.constant 5.000000e+00 : f64
// CHECK-NEXT:        %8 = arith.constant 6.000000e+00 : f64
// CHECK-NEXT:        %9 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<2x3xf64>
// CHECK-NEXT:        %10 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<2x3xf64>
// CHECK-NEXT:        "affine.store"(%3, %10) {"map" = affine_map<() -> (0, 0)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "affine.store"(%4, %10) {"map" = affine_map<() -> (0, 1)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "affine.store"(%5, %10) {"map" = affine_map<() -> (0, 2)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "affine.store"(%6, %10) {"map" = affine_map<() -> (1, 0)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "affine.store"(%7, %10) {"map" = affine_map<() -> (1, 1)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "affine.store"(%8, %10) {"map" = affine_map<() -> (1, 2)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "affine.store"(%3, %9) {"map" = affine_map<() -> (0, 0)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "affine.store"(%4, %9) {"map" = affine_map<() -> (0, 1)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "affine.store"(%5, %9) {"map" = affine_map<() -> (0, 2)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "affine.store"(%6, %9) {"map" = affine_map<() -> (1, 0)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "affine.store"(%7, %9) {"map" = affine_map<() -> (1, 1)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "affine.store"(%8, %9) {"map" = affine_map<() -> (1, 2)>} : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "toy_accelerator.transpose"(%2, %9) {"source_rows" = 2 : index, "source_cols" = 3 : index} : (memref<3x2xf64>, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "toy_accelerator.transpose"(%1, %10) {"source_rows" = 2 : index, "source_cols" = 3 : index} : (memref<3x2xf64>, memref<2x3xf64>) -> ()
// CHECK-NEXT:        "toy_accelerator.mul"(%0, %2, %1) : (memref<3x2xf64>, memref<3x2xf64>, memref<3x2xf64>) -> ()
// CHECK-NEXT:        printf.print_format "{}", %0 : memref<3x2xf64>
// CHECK-NEXT:        "memref.dealloc"(%10) : (memref<2x3xf64>) -> ()
// CHECK-NEXT:        "memref.dealloc"(%9) : (memref<2x3xf64>) -> ()
// CHECK-NEXT:        "memref.dealloc"(%2) : (memref<3x2xf64>) -> ()
// CHECK-NEXT:        "memref.dealloc"(%1) : (memref<3x2xf64>) -> ()
// CHECK-NEXT:        "memref.dealloc"(%0) : (memref<3x2xf64>) -> ()
// CHECK-NEXT:        func.return
// CHECK-NEXT:      }
// CHECK-NEXT:    }
