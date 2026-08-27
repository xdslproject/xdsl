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

// -----

// A tiled reduction dimension is absent from the output indexing map, so each
// tile takes the whole output, reads what the last tile left in it and
// accumulates into that. Here only the reduction dimension of a matmul is tiled.

%A, %B, %C = "test.op"() : () -> (tensor<4x6xf32>, tensor<6x4xf32>, tensor<4x4xf32>)

%D = linalg.generic {
  indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B : tensor<4x6xf32>, tensor<6x4xf32>) outs(%C : tensor<4x4xf32>) attrs = {test_tile_sizes = array<i32: 0, 0, 2>} {
^bb0(%a: f32, %b: f32, %c: f32):
  %m = arith.mulf %a, %b : f32
  %s = arith.addf %c, %m : f32
  linalg.yield %s : f32
} -> tensor<4x4xf32>

"test.op"(%D) : (tensor<4x4xf32>) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %B, %C = "test.op"() : () -> (tensor<4x6xf32>, tensor<6x4xf32>, tensor<4x4xf32>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 6 : index
// CHECK-NEXT:   %2 = arith.constant 2 : index
// CHECK-NEXT:   %D = scf.for %3 = %0 to %1 step %2 iter_args(%4 = %C) -> (tensor<4x4xf32>) {
// CHECK-NEXT:     %5 = tensor.extract_slice %A[0, %3] [4, 2] [1, 1] : tensor<4x6xf32> to tensor<4x2xf32>
// CHECK-NEXT:     %6 = tensor.extract_slice %B[%3, 0] [2, 4] [1, 1] : tensor<6x4xf32> to tensor<2x4xf32>
// CHECK-NEXT:     %7 = tensor.extract_slice %4[0, 0] [4, 4] [1, 1] : tensor<4x4xf32> to tensor<4x4xf32>
// CHECK-NEXT:     %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5, %6 : tensor<4x2xf32>, tensor<2x4xf32>) outs(%7 : tensor<4x4xf32>) {
// CHECK-NEXT:     ^bb0(%a: f32, %b: f32, %c: f32):
// CHECK-NEXT:       %m = arith.mulf %a, %b : f32
// CHECK-NEXT:       %s = arith.addf %c, %m : f32
// CHECK-NEXT:       linalg.yield %s : f32
// CHECK-NEXT:     } -> tensor<4x4xf32>
// CHECK-NEXT:     %9 = tensor.insert_slice %8 into %4[0, 0] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<4x4xf32>
// CHECK-NEXT:     scf.yield %9 : tensor<4x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   "test.op"(%D) : (tensor<4x4xf32>) -> ()
// CHECK-NEXT: }

// -----

// The same over memrefs, where a tile accumulates into the output through its
// subview rather than through a carried value.

%A, %B, %C = "test.op"() : () -> (memref<4x6xf32>, memref<6x4xf32>, memref<4x4xf32>)

linalg.generic {
  indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B : memref<4x6xf32>, memref<6x4xf32>) outs(%C : memref<4x4xf32>) attrs = {test_tile_sizes = array<i32: 0, 0, 2>} {
^bb0(%a: f32, %b: f32, %c: f32):
  %m = arith.mulf %a, %b : f32
  %s = arith.addf %c, %m : f32
  linalg.yield %s : f32
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %B, %C = "test.op"() : () -> (memref<4x6xf32>, memref<6x4xf32>, memref<4x4xf32>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 6 : index
// CHECK-NEXT:   %2 = arith.constant 2 : index
// CHECK-NEXT:   scf.for %3 = %0 to %1 step %2 {
// CHECK-NEXT:     %4 = memref.subview %A[0, %3] [4, 2] [1, 1] : memref<4x6xf32> to memref<4x2xf32, strided<[6, 1], offset: ?>>
// CHECK-NEXT:     %5 = memref.subview %B[%3, 0] [2, 4] [1, 1] : memref<6x4xf32> to memref<2x4xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:     %6 = memref.subview %C[0, 0] [4, 4] [1, 1] : memref<4x4xf32> to memref<4x4xf32, strided<[4, 1]>>
// CHECK-NEXT:     linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%4, %5 : memref<4x2xf32, strided<[6, 1], offset: ?>>, memref<2x4xf32, strided<[4, 1], offset: ?>>) outs(%6 : memref<4x4xf32, strided<[4, 1]>>) {
// CHECK-NEXT:     ^bb0(%a: f32, %b: f32, %c: f32):
// CHECK-NEXT:       %m = arith.mulf %a, %b : f32
// CHECK-NEXT:       %s = arith.addf %c, %m : f32
// CHECK-NEXT:       linalg.yield %s : f32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// A reduction dimension that the tile size does not divide, where the last tile
// runs past the end of the dimension and is clamped like any other.

%A, %B, %C = "test.op"() : () -> (tensor<4x6xf32>, tensor<6x4xf32>, tensor<4x4xf32>)

%D = linalg.generic {
  indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B : tensor<4x6xf32>, tensor<6x4xf32>) outs(%C : tensor<4x4xf32>) attrs = {test_tile_sizes = array<i32: 0, 0, 4>} {
^bb0(%a: f32, %b: f32, %c: f32):
  %m = arith.mulf %a, %b : f32
  %s = arith.addf %c, %m : f32
  linalg.yield %s : f32
} -> tensor<4x4xf32>

"test.op"(%D) : (tensor<4x4xf32>) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %B, %C = "test.op"() : () -> (tensor<4x6xf32>, tensor<6x4xf32>, tensor<4x4xf32>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 6 : index
// CHECK-NEXT:   %2 = arith.constant 4 : index
// CHECK-NEXT:   %D = scf.for %3 = %0 to %1 step %2 iter_args(%4 = %C) -> (tensor<4x4xf32>) {
// CHECK-NEXT:     %5 = affine.min affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))> (%2, %1, %3)
// CHECK-NEXT:     %6 = tensor.extract_slice %A[0, %3] [4, %5] [1, 1] : tensor<4x6xf32> to tensor<4x?xf32>
// CHECK-NEXT:     %7 = tensor.extract_slice %B[%3, 0] [%5, 4] [1, 1] : tensor<6x4xf32> to tensor<?x4xf32>
// CHECK-NEXT:     %8 = tensor.extract_slice %4[0, 0] [4, 4] [1, 1] : tensor<4x4xf32> to tensor<4x4xf32>
// CHECK-NEXT:     %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6, %7 : tensor<4x?xf32>, tensor<?x4xf32>) outs(%8 : tensor<4x4xf32>) {
// CHECK-NEXT:     ^bb0(%a: f32, %b: f32, %c: f32):
// CHECK-NEXT:       %m = arith.mulf %a, %b : f32
// CHECK-NEXT:       %s = arith.addf %c, %m : f32
// CHECK-NEXT:       linalg.yield %s : f32
// CHECK-NEXT:     } -> tensor<4x4xf32>
// CHECK-NEXT:     %10 = tensor.insert_slice %9 into %4[0, 0] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<4x4xf32>
// CHECK-NEXT:     scf.yield %10 : tensor<4x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   "test.op"(%D) : (tensor<4x4xf32>) -> ()
// CHECK-NEXT: }

// -----

// A linalg.index reads the position of the iteration it runs in, which inside a
// tile is a position within that tile, so the offset the tile starts at is added
// back to it. Here the first dimension is tiled and the second is not, so only
// the index of the first is offset.

%A, %C = "test.op"() : () -> (memref<4x6xindex>, memref<4x6xindex>)

linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]
} ins(%A : memref<4x6xindex>) outs(%C : memref<4x6xindex>) attrs = {test_tile_sizes = array<i32: 2, 0>} {
^bb0(%a: index, %c: index):
  %i = linalg.index 0 : index
  %j = linalg.index 1 : index
  %s = arith.addi %i, %j : index
  linalg.yield %s : index
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %C = "test.op"() : () -> (memref<4x6xindex>, memref<4x6xindex>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 4 : index
// CHECK-NEXT:   %2 = arith.constant 2 : index
// CHECK-NEXT:   scf.for %3 = %0 to %1 step %2 {
// CHECK-NEXT:     %4 = memref.subview %A[%3, 0] [2, 6] [1, 1] : memref<4x6xindex> to memref<2x6xindex, strided<[6, 1], offset: ?>>
// CHECK-NEXT:     %5 = memref.subview %C[%3, 0] [2, 6] [1, 1] : memref<4x6xindex> to memref<2x6xindex, strided<[6, 1], offset: ?>>
// CHECK-NEXT:     linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : memref<2x6xindex, strided<[6, 1], offset: ?>>) outs(%5 : memref<2x6xindex, strided<[6, 1], offset: ?>>) {
// CHECK-NEXT:     ^bb0(%a: index, %c: index):
// CHECK-NEXT:       %i = linalg.index 0 : index
// CHECK-NEXT:       %i_1 = affine.apply affine_map<(d0, d1) -> ((d0 + d1))> (%i, %3)
// CHECK-NEXT:       %j = linalg.index 1 : index
// CHECK-NEXT:       %s = arith.addi %i_1, %j : index
// CHECK-NEXT:       linalg.yield %s : index
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// An index read more than once takes the offset one at each of its readers, and
// a dimension whose tile size does not divide it is offset like any other.

%A, %C = "test.op"() : () -> (tensor<6xindex>, tensor<6xindex>)

%D = linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
  iterator_types = ["parallel"]
} ins(%A : tensor<6xindex>) outs(%C : tensor<6xindex>) attrs = {test_tile_sizes = array<i32: 4>} {
^bb0(%a: index, %c: index):
  %i = linalg.index 0 : index
  %s = arith.muli %i, %i : index
  linalg.yield %s : index
} -> tensor<6xindex>

"test.op"(%D) : (tensor<6xindex>) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %C = "test.op"() : () -> (tensor<6xindex>, tensor<6xindex>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 6 : index
// CHECK-NEXT:   %2 = arith.constant 4 : index
// CHECK-NEXT:   %D = scf.for %3 = %0 to %1 step %2 iter_args(%4 = %C) -> (tensor<6xindex>) {
// CHECK-NEXT:     %5 = affine.min affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))> (%2, %1, %3)
// CHECK-NEXT:     %6 = tensor.extract_slice %A[%3] [%5] [1] : tensor<6xindex> to tensor<?xindex>
// CHECK-NEXT:     %7 = tensor.extract_slice %4[%3] [%5] [1] : tensor<6xindex> to tensor<?xindex>
// CHECK-NEXT:     %8 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%6 : tensor<?xindex>) outs(%7 : tensor<?xindex>) {
// CHECK-NEXT:     ^bb0(%a: index, %c: index):
// CHECK-NEXT:       %i = linalg.index 0 : index
// CHECK-NEXT:       %i_1 = affine.apply affine_map<(d0, d1) -> ((d0 + d1))> (%i, %3)
// CHECK-NEXT:       %s = arith.muli %i_1, %i_1 : index
// CHECK-NEXT:       linalg.yield %s : index
// CHECK-NEXT:     } -> tensor<?xindex>
// CHECK-NEXT:     %9 = tensor.insert_slice %8 into %4[%3] [%5] [1] : tensor<?xindex> into tensor<6xindex>
// CHECK-NEXT:     scf.yield %9 : tensor<6xindex>
// CHECK-NEXT:   }
// CHECK-NEXT:   "test.op"(%D) : (tensor<6xindex>) -> ()
// CHECK-NEXT: }

// -----

// A named op is tiled into the same named op over the slices, rather than into
// a generic. What a linalg.matmul computes is which op it is, which tiling has
// no reason to take away from it.

%A, %B, %C = "test.op"() : () -> (memref<4x6xf32>, memref<6x4xf32>, memref<4x4xf32>)

linalg.matmul {test_tile_sizes = array<i32: 2, 2, 0>} ins(%A, %B : memref<4x6xf32>, memref<6x4xf32>) outs(%C : memref<4x4xf32>)

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %B, %C = "test.op"() : () -> (memref<4x6xf32>, memref<6x4xf32>, memref<4x4xf32>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 4 : index
// CHECK-NEXT:   %2 = arith.constant 4 : index
// CHECK-NEXT:   %3 = arith.constant 2 : index
// CHECK-NEXT:   %4 = arith.constant 2 : index
// CHECK-NEXT:   scf.for %5 = %0 to %1 step %3 {
// CHECK-NEXT:     scf.for %6 = %0 to %2 step %4 {
// CHECK-NEXT:       %7 = memref.subview %A[%5, 0] [2, 6] [1, 1] : memref<4x6xf32> to memref<2x6xf32, strided<[6, 1], offset: ?>>
// CHECK-NEXT:       %8 = memref.subview %B[0, %6] [6, 2] [1, 1] : memref<6x4xf32> to memref<6x2xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:       %9 = memref.subview %C[%5, %6] [2, 2] [1, 1] : memref<4x4xf32> to memref<2x2xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:       linalg.matmul ins(%7, %8 : memref<2x6xf32, strided<[6, 1], offset: ?>>, memref<6x2xf32, strided<[4, 1], offset: ?>>) outs(%9 : memref<2x2xf32, strided<[4, 1], offset: ?>>)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// The same over tensors, tiled over the dimension the matmul reduces, so that
// the tiles accumulate into the value the loop carries.

%A, %B, %C = "test.op"() : () -> (tensor<4x6xf32>, tensor<6x4xf32>, tensor<4x4xf32>)

%D = linalg.matmul {test_tile_sizes = array<i32: 0, 0, 2>} ins(%A, %B : tensor<4x6xf32>, tensor<6x4xf32>) outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>

"test.op"(%D) : (tensor<4x4xf32>) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %B, %C = "test.op"() : () -> (tensor<4x6xf32>, tensor<6x4xf32>, tensor<4x4xf32>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 6 : index
// CHECK-NEXT:   %2 = arith.constant 2 : index
// CHECK-NEXT:   %D = scf.for %3 = %0 to %1 step %2 iter_args(%4 = %C) -> (tensor<4x4xf32>) {
// CHECK-NEXT:     %5 = tensor.extract_slice %A[0, %3] [4, 2] [1, 1] : tensor<4x6xf32> to tensor<4x2xf32>
// CHECK-NEXT:     %6 = tensor.extract_slice %B[%3, 0] [2, 4] [1, 1] : tensor<6x4xf32> to tensor<2x4xf32>
// CHECK-NEXT:     %7 = tensor.extract_slice %4[0, 0] [4, 4] [1, 1] : tensor<4x4xf32> to tensor<4x4xf32>
// CHECK-NEXT:     %8 = linalg.matmul ins(%5, %6 : tensor<4x2xf32>, tensor<2x4xf32>) outs(%7 : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:     %9 = tensor.insert_slice %8 into %4[0, 0] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<4x4xf32>
// CHECK-NEXT:     scf.yield %9 : tensor<4x4xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   "test.op"(%D) : (tensor<4x4xf32>) -> ()
// CHECK-NEXT: }

// -----

// An indexing map result that reads more than one loop, so that the operand
// dimension it addresses is not one loop's to tile. It is read from where the
// tiled loop is and spans the loops it reads together, the tile of 2 in the
// first and the whole 4 of the second reaching 5 elements.

%A, %C = "test.op"() : () -> (memref<8xf32>, memref<4x4xf32>)

linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0 + d1)>, affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]
} ins(%A : memref<8xf32>) outs(%C : memref<4x4xf32>) attrs = {test_tile_sizes = array<i32: 2, 0>} {
^bb0(%a: f32, %c: f32):
  linalg.yield %a : f32
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %C = "test.op"() : () -> (memref<8xf32>, memref<4x4xf32>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 4 : index
// CHECK-NEXT:   %2 = arith.constant 2 : index
// CHECK-NEXT:   scf.for %3 = %0 to %1 step %2 {
// CHECK-NEXT:     %4 = memref.subview %A[%3] [5] [1] : memref<8xf32> to memref<5xf32, strided<[1], offset: ?>>
// CHECK-NEXT:     %5 = memref.subview %C[%3, 0] [2, 4] [1, 1] : memref<4x4xf32> to memref<2x4xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:     linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ((d0 + d1))>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : memref<5xf32, strided<[1], offset: ?>>) outs(%5 : memref<2x4xf32, strided<[4, 1], offset: ?>>) {
// CHECK-NEXT:     ^bb0(%a: f32, %c: f32):
// CHECK-NEXT:       linalg.yield %a : f32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// A result that steps through its operand rather than walking it, and one that
// is offset by a constant. The stride is left at one and taken up by the size,
// which spans what the tile reaches, and the constant drops out of the offset,
// which the operand is not moved along by.

%A, %B, %C = "test.op"() : () -> (memref<16xf32>, memref<16xf32>, memref<8xf32>)

linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0 * 2)>, affine_map<(d0) -> (d0 + 3)>, affine_map<(d0) -> (d0)>],
  iterator_types = ["parallel"]
} ins(%A, %B : memref<16xf32>, memref<16xf32>) outs(%C : memref<8xf32>) attrs = {test_tile_sizes = array<i32: 4>} {
^bb0(%a: f32, %b: f32, %c: f32):
  %s = arith.addf %a, %b : f32
  linalg.yield %s : f32
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %B, %C = "test.op"() : () -> (memref<16xf32>, memref<16xf32>, memref<8xf32>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 8 : index
// CHECK-NEXT:   %2 = arith.constant 4 : index
// CHECK-NEXT:   scf.for %3 = %0 to %1 step %2 {
// CHECK-NEXT:     %4 = affine.apply affine_map<(d0) -> ((d0 * 2))> (%3)
// CHECK-NEXT:     %5 = memref.subview %A[%4] [7] [1] : memref<16xf32> to memref<7xf32, strided<[1], offset: ?>>
// CHECK-NEXT:     %6 = memref.subview %B[%3] [7] [1] : memref<16xf32> to memref<7xf32, strided<[1], offset: ?>>
// CHECK-NEXT:     %7 = memref.subview %C[%3] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-NEXT:     linalg.generic {indexing_maps = [affine_map<(d0) -> ((d0 * 2))>, affine_map<(d0) -> ((d0 + 3))>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%5, %6 : memref<7xf32, strided<[1], offset: ?>>, memref<7xf32, strided<[1], offset: ?>>) outs(%7 : memref<4xf32, strided<[1], offset: ?>>) {
// CHECK-NEXT:     ^bb0(%a: f32, %b: f32, %c: f32):
// CHECK-NEXT:       %s = arith.addf %a, %b : f32
// CHECK-NEXT:       linalg.yield %s : f32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// One operand read two ways at once, where a tile size divides one loop and not
// the other. The first dimension is read from both loops and spans a tile of
// each, which the loop it does not divide leaves running to a size worked out as
// the op runs. The second is read from the loop that is divided alone, so its
// span is known here and no work is left for the op to do.

%A, %C = "test.op"() : () -> (memref<9x7xf32>, memref<6x4xf32>)

linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0 + d1, d1 * 2)>, affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]
} ins(%A : memref<9x7xf32>) outs(%C : memref<6x4xf32>) attrs = {test_tile_sizes = array<i32: 4, 2>} {
^bb0(%a: f32, %c: f32):
  linalg.yield %a : f32
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %A, %C = "test.op"() : () -> (memref<9x7xf32>, memref<6x4xf32>)
// CHECK-NEXT:   %0 = arith.constant 0 : index
// CHECK-NEXT:   %1 = arith.constant 6 : index
// CHECK-NEXT:   %2 = arith.constant 4 : index
// CHECK-NEXT:   %3 = arith.constant 4 : index
// CHECK-NEXT:   %4 = arith.constant 2 : index
// CHECK-NEXT:   scf.for %5 = %0 to %1 step %3 {
// CHECK-NEXT:     scf.for %6 = %0 to %2 step %4 {
// CHECK-NEXT:       %7 = affine.min affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))> (%3, %1, %5)
// CHECK-NEXT:       %8 = affine.apply affine_map<(d0) -> ((d0 + -1))> (%7)
// CHECK-NEXT:       %9 = affine.apply affine_map<(d0, d1) -> ((d0 + d1))> (%5, %6)
// CHECK-NEXT:       %10 = affine.apply affine_map<(d0) -> ((d0 + 2))> (%8)
// CHECK-NEXT:       %11 = affine.apply affine_map<(d0, d1) -> ((d1 * 2))> (%5, %6)
// CHECK-NEXT:       %12 = memref.subview %A[%9, %11] [%10, 3] [1, 1] : memref<9x7xf32> to memref<?x3xf32, strided<[7, 1], offset: ?>>
// CHECK-NEXT:       %13 = memref.subview %C[%5, %6] [%7, 2] [1, 1] : memref<6x4xf32> to memref<?x2xf32, strided<[4, 1], offset: ?>>
// CHECK-NEXT:       linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ((d0 + d1), (d1 * 2))>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12 : memref<?x3xf32, strided<[7, 1], offset: ?>>) outs(%13 : memref<?x2xf32, strided<[4, 1], offset: ?>>) {
// CHECK-NEXT:       ^bb0(%a: f32, %c: f32):
// CHECK-NEXT:         linalg.yield %a : f32
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
