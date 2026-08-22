// RUN: xdsl-opt %s --split-input-file -p test-linalg-tiling | filecheck %s

%A, %B, %C = "test.op"() : () -> (memref<4x4xf64>, memref<4x4xf64>, memref<4x4xf64>)

linalg.generic {
    indexing_maps = [
        affine_map<(i, j) -> (i, j)>,
        affine_map<(i, j) -> (i, j)>,
        affine_map<(i, j) -> (i, j)>
    ],
    iterator_types = ["parallel", "parallel"]
} ins(%A, %B : memref<4x4xf64>, memref<4x4xf64>) outs(%C : memref<4x4xf64>) attrs = {test_tile_sizes = array<i32: 2, 2>} {
^bb0(%a: f64, %b: f64, %c: f64):
    %sum = arith.addf %a, %b : f64
    linalg.yield %sum : f64
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %B, %C = "test.op"() : () -> (memref<4x4xf64>, memref<4x4xf64>, memref<4x4xf64>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 4 : index
// CHECK-NEXT:   %2 = arith.constant 4 : index
// CHECK-NEXT:   %3 = arith.constant 2 : index
// CHECK-NEXT:   %4 = arith.constant 2 : index
// CHECK-NEXT:   scf.for %5 = %0 to %1 step %3 {
// CHECK-NEXT:     scf.for %6 = %0 to %2 step %4 {
// CHECK-NEXT:       %7 = memref.subview %A[%5, %6] [2, 2] [1, 1] : memref<4x4xf64> to memref<2x2xf64, strided<[4, 1], offset: ?>>
// CHECK-NEXT:       %8 = memref.subview %B[%5, %6] [2, 2] [1, 1] : memref<4x4xf64> to memref<2x2xf64, strided<[4, 1], offset: ?>>
// CHECK-NEXT:       %9 = memref.subview %C[%5, %6] [2, 2] [1, 1] : memref<4x4xf64> to memref<2x2xf64, strided<[4, 1], offset: ?>>
// CHECK-NEXT:       linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7, %8 : memref<2x2xf64, strided<[4, 1], offset: ?>>, memref<2x2xf64, strided<[4, 1], offset: ?>>) outs(%9 : memref<2x2xf64, strided<[4, 1], offset: ?>>) {
// CHECK-NEXT:       ^bb0(%a: f64, %b: f64, %c: f64):
// CHECK-NEXT:         %sum = arith.addf %a, %b : f64
// CHECK-NEXT:         linalg.yield %sum : f64
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

%A, %B, %C = "test.op"() : () -> (memref<4x4xf64>, memref<4x4xf64>, memref<4x4xf64>)

linalg.generic {
    indexing_maps = [
        affine_map<(i, j) -> (i, j)>,
        affine_map<(i, j) -> (i, j)>,
        affine_map<(i, j) -> (i, j)>
    ],
    iterator_types = ["parallel", "parallel"]
} ins(%A, %B : memref<4x4xf64>, memref<4x4xf64>) outs(%C : memref<4x4xf64>) attrs = {test_tile_sizes = array<i32: 2, 0>} {
^bb0(%a: f64, %b: f64, %c: f64):
    %sum = arith.addf %a, %b : f64
    linalg.yield %sum : f64
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %B, %C = "test.op"() : () -> (memref<4x4xf64>, memref<4x4xf64>, memref<4x4xf64>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 4 : index
// CHECK-NEXT:   %2 = arith.constant 2 : index
// CHECK-NEXT:   scf.for %3 = %0 to %1 step %2 {
// CHECK-NEXT:     %4 = memref.subview %A[%3, 0] [2, 4] [1, 1] : memref<4x4xf64> to memref<2x4xf64, strided<[4, 1], offset: ?>>
// CHECK-NEXT:     %5 = memref.subview %B[%3, 0] [2, 4] [1, 1] : memref<4x4xf64> to memref<2x4xf64, strided<[4, 1], offset: ?>>
// CHECK-NEXT:     %6 = memref.subview %C[%3, 0] [2, 4] [1, 1] : memref<4x4xf64> to memref<2x4xf64, strided<[4, 1], offset: ?>>
// CHECK-NEXT:     linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4, %5 : memref<2x4xf64, strided<[4, 1], offset: ?>>, memref<2x4xf64, strided<[4, 1], offset: ?>>) outs(%6 : memref<2x4xf64, strided<[4, 1], offset: ?>>) {
// CHECK-NEXT:     ^bb0(%a: f64, %b: f64, %c: f64):
// CHECK-NEXT:       %sum = arith.addf %a, %b : f64
// CHECK-NEXT:       linalg.yield %sum : f64
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// Dynamic source strides are preserved in tiled subviews.

%A, %B = "test.op"() : () -> (memref<4x4xf32, strided<[?, 1]>>, memref<4x4xf32, strided<[?, 1]>>)

linalg.generic {
    indexing_maps = [
        affine_map<(i, j) -> (i, j)>,
        affine_map<(i, j) -> (i, j)>
    ],
    iterator_types = ["parallel", "parallel"]
} ins(%A : memref<4x4xf32, strided<[?, 1]>>) outs(%B : memref<4x4xf32, strided<[?, 1]>>) attrs = {test_tile_sizes = array<i32: 2, 2>} {
^bb0(%a: f32, %b: f32):
    linalg.yield %a : f32
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %B = "test.op"() : () -> (memref<4x4xf32, strided<[?, 1]>>, memref<4x4xf32, strided<[?, 1]>>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 4 : index
// CHECK-NEXT:   %2 = arith.constant 4 : index
// CHECK-NEXT:   %3 = arith.constant 2 : index
// CHECK-NEXT:   %4 = arith.constant 2 : index
// CHECK-NEXT:   scf.for %5 = %0 to %1 step %3 {
// CHECK-NEXT:     scf.for %6 = %0 to %2 step %4 {
// CHECK-NEXT:       %7 = memref.subview %A[%5, %6] [2, 2] [1, 1] : memref<4x4xf32, strided<[?, 1]>> to memref<2x2xf32, strided<[?, 1], offset: ?>>
// CHECK-NEXT:       %8 = memref.subview %B[%5, %6] [2, 2] [1, 1] : memref<4x4xf32, strided<[?, 1]>> to memref<2x2xf32, strided<[?, 1], offset: ?>>
// CHECK-NEXT:       linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : memref<2x2xf32, strided<[?, 1], offset: ?>>) outs(%8 : memref<2x2xf32, strided<[?, 1], offset: ?>>) {
// CHECK-NEXT:       ^bb0(%a: f32, %b: f32):
// CHECK-NEXT:         linalg.yield %a : f32
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// Tensors are tiled by extracting each tile, computing it, and inserting the
// result back into the tensor carried by the loops.

%A, %B = "test.op"() : () -> (tensor<4x4xf32>, tensor<4x4xf32>)

%C = "linalg.generic"(%A, %B) <{
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
  operandSegmentSizes = array<i32: 1, 1>
}> ({
^bb0(%a: f32, %b: f32):
  "linalg.yield"(%a) : (f32) -> ()
}) {test_tile_sizes = array<i32: 2, 2>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>

"test.op"(%C) : (tensor<4x4xf32>) -> ()

// Each loop carries the tensor: the outer one from %B, the inner one from the
// block argument of the outer one. The output tile is extracted from the
// carried value, not from %B, so tiles accumulate.

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %B = "test.op"() : () -> (tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 4 : index
// CHECK-NEXT:   %2 = arith.constant 4 : index
// CHECK-NEXT:   %3 = arith.constant 2 : index
// CHECK-NEXT:   %4 = arith.constant 2 : index
// CHECK-NEXT:   %C = scf.for %5 = %0 to %1 step %3 iter_args(%6 = %B) -> (tensor<4x4xf32>) {
// CHECK-NEXT:     %7 = scf.for %8 = %0 to %2 step %4 iter_args(%9 = %6) -> (tensor<4x4xf32>) {
// CHECK-NEXT:       %10 = tensor.extract_slice %A[%5, %8] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
// CHECK-NEXT:       %11 = tensor.extract_slice %9[%5, %8] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
// CHECK-NEXT:       %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<2x2xf32>) outs(%11 : tensor<2x2xf32>) {
// CHECK-NEXT:       ^bb0(%a: f32, %b: f32):
// CHECK-NEXT:         linalg.yield %a : f32
// CHECK-NEXT:       } -> tensor<2x2xf32>
// CHECK-NEXT:       %13 = tensor.insert_slice %12 into %9[%5, %8] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<4x4xf32>
// CHECK-NEXT:       scf.yield %13 : tensor<4x4xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %7 : tensor<4x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   "test.op"(%C) : (tensor<4x4xf32>) -> ()
// CHECK-NEXT: }

// -----

// An untiled tensor dimension takes its whole extent.

%A, %B = "test.op"() : () -> (tensor<4x4xf32>, tensor<4x4xf32>)

%C = "linalg.generic"(%A, %B) <{
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
  operandSegmentSizes = array<i32: 1, 1>
}> ({
^bb0(%a: f32, %b: f32):
  "linalg.yield"(%a) : (f32) -> ()
}) {test_tile_sizes = array<i32: 2, 0>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>

"test.op"(%C) : (tensor<4x4xf32>) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %B = "test.op"() : () -> (tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 4 : index
// CHECK-NEXT:   %2 = arith.constant 2 : index
// CHECK-NEXT:   %C = scf.for %3 = %0 to %1 step %2 iter_args(%4 = %B) -> (tensor<4x4xf32>) {
// CHECK-NEXT:     %5 = tensor.extract_slice %A[%3, 0] [2, 4] [1, 1] : tensor<4x4xf32> to tensor<2x4xf32>
// CHECK-NEXT:     %6 = tensor.extract_slice %4[%3, 0] [2, 4] [1, 1] : tensor<4x4xf32> to tensor<2x4xf32>
// CHECK-NEXT:     %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<2x4xf32>) outs(%6 : tensor<2x4xf32>) {
// CHECK-NEXT:     ^bb0(%a: f32, %b: f32):
// CHECK-NEXT:       linalg.yield %a : f32
// CHECK-NEXT:     } -> tensor<2x4xf32>
// CHECK-NEXT:     %8 = tensor.insert_slice %7 into %4[%3, 0] [2, 4] [1, 1] : tensor<2x4xf32> into tensor<4x4xf32>
// CHECK-NEXT:     scf.yield %8 : tensor<4x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   "test.op"(%C) : (tensor<4x4xf32>) -> ()
// CHECK-NEXT: }

// -----

// A dimension whose range does not divide by its tile size gets a smaller
// tile on its last iteration, sized with affine.min. The other dimension
// divides evenly and keeps its static tile size.

%A, %B = "test.op"() : () -> (memref<5x4xf32>, memref<5x4xf32>)
linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]
} ins(%A : memref<5x4xf32>) outs(%B : memref<5x4xf32>) attrs = {test_tile_sizes = array<i32: 2, 2>} {
^bb0(%a: f32, %b: f32):
    linalg.yield %a : f32
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %B = "test.op"() : () -> (memref<5x4xf32>, memref<5x4xf32>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 5 : index
// CHECK-NEXT:   %2 = arith.constant 4 : index
// CHECK-NEXT:   %3 = arith.constant 2 : index
// CHECK-NEXT:   %4 = arith.constant 2 : index
// CHECK-NEXT:   scf.for %5 = %0 to %1 step %3 {
// CHECK-NEXT:     scf.for %6 = %0 to %2 step %4 {
// CHECK-NEXT:       %7 = affine.min affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))> (%3, %1, %5)
// CHECK-NEXT:       %8 = memref.subview %A[%5, %6] [%7, 2] [1, 1] : memref<5x4xf32> to memref<?x2xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:       %9 = memref.subview %B[%5, %6] [%7, 2] [1, 1] : memref<5x4xf32> to memref<?x2xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:       linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8 : memref<?x2xf32, strided<[4, 1], offset: ?>>) outs(%9 : memref<?x2xf32, strided<[4, 1], offset: ?>>) {
// CHECK-NEXT:       ^bb0(%a: f32, %b: f32):
// CHECK-NEXT:         linalg.yield %a : f32
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// A leftover tile on a tensor is extracted, computed and written back at the
// same size, since the extract and the insert share slice parameters.

%A, %B = "test.op"() : () -> (tensor<5x4xf32>, tensor<5x4xf32>)
%C = "linalg.generic"(%A, %B) <{
  indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
  iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
  operandSegmentSizes = array<i32: 1, 1>
}> ({
^bb0(%a: f32, %b: f32):
  "linalg.yield"(%a) : (f32) -> ()
}) {test_tile_sizes = array<i32: 2, 2>} : (tensor<5x4xf32>, tensor<5x4xf32>) -> tensor<5x4xf32>
"test.op"(%C) : (tensor<5x4xf32>) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %B = "test.op"() : () -> (tensor<5x4xf32>, tensor<5x4xf32>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 5 : index
// CHECK-NEXT:   %2 = arith.constant 4 : index
// CHECK-NEXT:   %3 = arith.constant 2 : index
// CHECK-NEXT:   %4 = arith.constant 2 : index
// CHECK-NEXT:   %C = scf.for %5 = %0 to %1 step %3 iter_args(%6 = %B) -> (tensor<5x4xf32>) {
// CHECK-NEXT:     %7 = scf.for %8 = %0 to %2 step %4 iter_args(%9 = %6) -> (tensor<5x4xf32>) {
// CHECK-NEXT:       %10 = affine.min affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))> (%3, %1, %5)
// CHECK-NEXT:       %11 = tensor.extract_slice %A[%5, %8] [%10, 2] [1, 1] : tensor<5x4xf32> to tensor<?x2xf32>
// CHECK-NEXT:       %12 = tensor.extract_slice %9[%5, %8] [%10, 2] [1, 1] : tensor<5x4xf32> to tensor<?x2xf32>
// CHECK-NEXT:       %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<?x2xf32>) outs(%12 : tensor<?x2xf32>) {
// CHECK-NEXT:       ^bb0(%a: f32, %b: f32):
// CHECK-NEXT:         linalg.yield %a : f32
// CHECK-NEXT:       } -> tensor<?x2xf32>
// CHECK-NEXT:       %14 = tensor.insert_slice %13 into %9[%5, %8] [%10, 2] [1, 1] : tensor<?x2xf32> into tensor<5x4xf32>
// CHECK-NEXT:       scf.yield %14 : tensor<5x4xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %7 : tensor<5x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   "test.op"(%C) : (tensor<5x4xf32>) -> ()
// CHECK-NEXT: }

// -----

// A loop range that is not known until the op runs is read back off an
// operand, and its tiles are clamped like any other leftover.

%A, %B = "test.op"() : () -> (memref<?x4xf32>, memref<?x4xf32>)
linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]
} ins(%A : memref<?x4xf32>) outs(%B : memref<?x4xf32>) attrs = {test_tile_sizes = array<i32: 2, 0>} {
^bb0(%a: f32, %b: f32):
    linalg.yield %a : f32
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %B = "test.op"() : () -> (memref<?x4xf32>, memref<?x4xf32>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 0 : index
// CHECK-NEXT:   %2 = memref.dim %A, %1 : memref<?x4xf32>
// CHECK-NEXT:   %3 = arith.constant 2 : index
// CHECK-NEXT:   scf.for %4 = %0 to %2 step %3 {
// CHECK-NEXT:     %5 = affine.min affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))> (%3, %2, %4)
// CHECK-NEXT:     %6 = memref.subview %A[%4, 0] [%5, 4] [1, 1] : memref<?x4xf32> to memref<?x4xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:     %7 = memref.subview %B[%4, 0] [%5, 4] [1, 1] : memref<?x4xf32> to memref<?x4xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:     linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6 : memref<?x4xf32, strided<[4, 1], offset: ?>>) outs(%7 : memref<?x4xf32, strided<[4, 1], offset: ?>>) {
// CHECK-NEXT:     ^bb0(%a: f32, %b: f32):
// CHECK-NEXT:       linalg.yield %a : f32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// The same for tensors, read with tensor.dim.

%A, %B = "test.op"() : () -> (tensor<?x4xf32>, tensor<?x4xf32>)
%C = "linalg.generic"(%A, %B) <{
  indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
  iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
  operandSegmentSizes = array<i32: 1, 1>
}> ({
^bb0(%a: f32, %b: f32):
  "linalg.yield"(%a) : (f32) -> ()
}) {test_tile_sizes = array<i32: 2, 0>} : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
"test.op"(%C) : (tensor<?x4xf32>) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %B = "test.op"() : () -> (tensor<?x4xf32>, tensor<?x4xf32>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 0 : index
// CHECK-NEXT:   %2 = tensor.dim %A, %1 : tensor<?x4xf32>
// CHECK-NEXT:   %3 = arith.constant 2 : index
// CHECK-NEXT:   %C = scf.for %4 = %0 to %2 step %3 iter_args(%5 = %B) -> (tensor<?x4xf32>) {
// CHECK-NEXT:     %6 = affine.min affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))> (%3, %2, %4)
// CHECK-NEXT:     %7 = tensor.extract_slice %A[%4, 0] [%6, 4] [1, 1] : tensor<?x4xf32> to tensor<?x4xf32>
// CHECK-NEXT:     %8 = tensor.extract_slice %5[%4, 0] [%6, 4] [1, 1] : tensor<?x4xf32> to tensor<?x4xf32>
// CHECK-NEXT:     %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?x4xf32>) outs(%8 : tensor<?x4xf32>) {
// CHECK-NEXT:     ^bb0(%a: f32, %b: f32):
// CHECK-NEXT:       linalg.yield %a : f32
// CHECK-NEXT:     } -> tensor<?x4xf32>
// CHECK-NEXT:     %10 = tensor.insert_slice %9 into %5[%4, 0] [%6, 4] [1, 1] : tensor<?x4xf32> into tensor<?x4xf32>
// CHECK-NEXT:     scf.yield %10 : tensor<?x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   "test.op"(%C) : (tensor<?x4xf32>) -> ()
// CHECK-NEXT: }

// -----

// A tile size that is not known until the op runs steps the loop directly.
// It cannot be shown to divide the range either, so its tiles are clamped.

%A, %B = "test.op"() : () -> (memref<8x4xf32>, memref<8x4xf32>)
linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]
} ins(%A : memref<8x4xf32>) outs(%B : memref<8x4xf32>) attrs = {test_tile_sizes = array<i32: 0, 0>, test_dynamic_tile_sizes = array<i32: 0>} {
^bb0(%a: f32, %b: f32):
    linalg.yield %a : f32
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %B = "test.op"() : () -> (memref<8x4xf32>, memref<8x4xf32>)
// CHECK-NEXT:   %0 = "test.op"() : () -> index
// CHECK-NEXT:   %1 = arith.constant 0 : index
// CHECK-NEXT:   %2 = arith.constant 8 : index
// CHECK-NEXT:   scf.for %3 = %1 to %2 step %0 {
// CHECK-NEXT:     %4 = affine.min affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))> (%0, %2, %3)
// CHECK-NEXT:     %5 = memref.subview %A[%3, 0] [%4, 4] [1, 1] : memref<8x4xf32> to memref<?x4xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:     %6 = memref.subview %B[%3, 0] [%4, 4] [1, 1] : memref<8x4xf32> to memref<?x4xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:     linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5 : memref<?x4xf32, strided<[4, 1], offset: ?>>) outs(%6 : memref<?x4xf32, strided<[4, 1], offset: ?>>) {
// CHECK-NEXT:     ^bb0(%a: f32, %b: f32):
// CHECK-NEXT:       linalg.yield %a : f32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
