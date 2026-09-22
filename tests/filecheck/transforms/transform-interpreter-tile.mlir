// RUN: xdsl-opt %s -p transform-interpreter --split-input-file | filecheck %s

// Zero sizes skip dimensions. Reuse the returned tiled-op handle to tile again,
// adding a loop around the smaller matmul. Check the resulting loops and slices.
module attributes {transform.with_named_sequence} {
  func.func @partial(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %D = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %D : tensor<4x4xf32>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %outer, %inner = transform.structured.tile_using_for %target tile_sizes [2, 0, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %retiled, %loop = transform.structured.tile_using_for %tiled tile_sizes [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK:       builtin.module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @partial(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:      %0 = arith.constant 0 : index
// CHECK-NEXT:      %1 = arith.constant 4 : index
// CHECK-NEXT:      %2 = arith.constant 4 : index
// CHECK-NEXT:      %3 = arith.constant 2 : index
// CHECK-NEXT:      %4 = arith.constant 2 : index
// CHECK-NEXT:      %D = scf.for %5 = %0 to %1 step %3 iter_args(%6 = %C) -> (tensor<4x4xf32>) {
// CHECK-NEXT:        %7 = scf.for %8 = %0 to %2 step %4 iter_args(%9 = %6) -> (tensor<4x4xf32>) {
// CHECK-NEXT:          %10 = tensor.extract_slice %A[%5, %8] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
// CHECK-NEXT:          %11 = tensor.extract_slice %B[%8, 0] [2, 4] [1, 1] : tensor<4x4xf32> to tensor<2x4xf32>
// CHECK-NEXT:          %12 = tensor.extract_slice %9[%5, 0] [2, 4] [1, 1] : tensor<4x4xf32> to tensor<2x4xf32>
// CHECK-NEXT:          %13 = arith.constant 0 : index
// CHECK-NEXT:          %14 = arith.constant 2 : index
// CHECK-NEXT:          %15 = arith.constant 1 : index
// CHECK-NEXT:          %16 = scf.for %17 = %13 to %14 step %15 iter_args(%18 = %12) -> (tensor<2x4xf32>) {
// CHECK-NEXT:            %19 = tensor.extract_slice %10[%17, 0] [1, 2] [1, 1] : tensor<2x2xf32> to tensor<1x2xf32>
// CHECK-NEXT:            %20 = tensor.extract_slice %11[0, 0] [2, 4] [1, 1] : tensor<2x4xf32> to tensor<2x4xf32>
// CHECK-NEXT:            %21 = tensor.extract_slice %18[%17, 0] [1, 4] [1, 1] : tensor<2x4xf32> to tensor<1x4xf32>
// CHECK-NEXT:            %22 = linalg.matmul ins(%19, %20 : tensor<1x2xf32>, tensor<2x4xf32>) outs(%21 : tensor<1x4xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:            %23 = tensor.insert_slice %22 into %18[%17, 0] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<2x4xf32>
// CHECK-NEXT:            scf.yield %23 : tensor<2x4xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:          %24 = tensor.insert_slice %16 into %9[%5, 0] [2, 4] [1, 1] : tensor<2x4xf32> into tensor<4x4xf32>
// CHECK-NEXT:          scf.yield %24 : tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %7 : tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %D : tensor<4x4xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %target = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled, %outer, %inner = transform.structured.tile_using_for %target tile_sizes [2, 0, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
// CHECK-NEXT:      %retiled, %loop = transform.structured.tile_using_for %tiled tile_sizes [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.yield
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

// A handle with two targets tiles both matmuls. Check the loops, slices, and
// accumulator threading in both functions.
module attributes {transform.with_named_sequence} {
  func.func @first(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %D = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %D : tensor<4x4xf32>
  }
  func.func @second(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %D = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %D : tensor<4x4xf32>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %targets = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %i, %j, %k = transform.structured.tile_using_for %targets tile_sizes [2, 2, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK:       builtin.module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @first(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:      %0 = arith.constant 0 : index
// CHECK-NEXT:      %1 = arith.constant 4 : index
// CHECK-NEXT:      %2 = arith.constant 4 : index
// CHECK-NEXT:      %3 = arith.constant 4 : index
// CHECK-NEXT:      %4 = arith.constant 2 : index
// CHECK-NEXT:      %5 = arith.constant 2 : index
// CHECK-NEXT:      %6 = arith.constant 2 : index
// CHECK-NEXT:      %D = scf.for %7 = %0 to %1 step %4 iter_args(%8 = %C) -> (tensor<4x4xf32>) {
// CHECK-NEXT:        %9 = scf.for %10 = %0 to %2 step %5 iter_args(%11 = %8) -> (tensor<4x4xf32>) {
// CHECK-NEXT:          %12 = scf.for %13 = %0 to %3 step %6 iter_args(%14 = %11) -> (tensor<4x4xf32>) {
// CHECK-NEXT:            %15 = tensor.extract_slice %A[%7, %13] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
// CHECK-NEXT:            %16 = tensor.extract_slice %B[%13, %10] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
// CHECK-NEXT:            %17 = tensor.extract_slice %14[%7, %10] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
// CHECK-NEXT:            %18 = linalg.matmul ins(%15, %16 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%17 : tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:            %19 = tensor.insert_slice %18 into %14[%7, %10] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<4x4xf32>
// CHECK-NEXT:            scf.yield %19 : tensor<4x4xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %12 : tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %9 : tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %D : tensor<4x4xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @second(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:      %0 = arith.constant 0 : index
// CHECK-NEXT:      %1 = arith.constant 4 : index
// CHECK-NEXT:      %2 = arith.constant 4 : index
// CHECK-NEXT:      %3 = arith.constant 4 : index
// CHECK-NEXT:      %4 = arith.constant 2 : index
// CHECK-NEXT:      %5 = arith.constant 2 : index
// CHECK-NEXT:      %6 = arith.constant 2 : index
// CHECK-NEXT:      %D = scf.for %7 = %0 to %1 step %4 iter_args(%8 = %C) -> (tensor<4x4xf32>) {
// CHECK-NEXT:        %9 = scf.for %10 = %0 to %2 step %5 iter_args(%11 = %8) -> (tensor<4x4xf32>) {
// CHECK-NEXT:          %12 = scf.for %13 = %0 to %3 step %6 iter_args(%14 = %11) -> (tensor<4x4xf32>) {
// CHECK-NEXT:            %15 = tensor.extract_slice %A[%7, %13] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
// CHECK-NEXT:            %16 = tensor.extract_slice %B[%13, %10] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
// CHECK-NEXT:            %17 = tensor.extract_slice %14[%7, %10] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
// CHECK-NEXT:            %18 = linalg.matmul ins(%15, %16 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%17 : tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:            %19 = tensor.insert_slice %18 into %14[%7, %10] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<4x4xf32>
// CHECK-NEXT:            scf.yield %19 : tensor<4x4xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %12 : tensor<4x4xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %9 : tensor<4x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %D : tensor<4x4xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %targets = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled, %i, %j, %k = transform.structured.tile_using_for %targets tile_sizes [2, 2, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.yield
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

// An empty input leaves the payload unchanged. Reusing each returned handle
// for tiling checks that all four results are empty, including the loop handles.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %empty = transform.structured.match ops{[]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %i, %j, %k = transform.structured.tile_using_for %empty tile_sizes [2, 2, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_again = transform.structured.tile_using_for %tiled tile_sizes [] : (!transform.any_op) -> !transform.any_op
    %i_again = transform.structured.tile_using_for %i tile_sizes [] : (!transform.any_op) -> !transform.any_op
    %j_again = transform.structured.tile_using_for %j tile_sizes [] : (!transform.any_op) -> !transform.any_op
    %k_again = transform.structured.tile_using_for %k tile_sizes [] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK:       builtin.module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %empty = transform.structured.match ops{[]} in %root : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled, %i, %j, %k = transform.structured.tile_using_for %empty tile_sizes [2, 2, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
// CHECK-NEXT:      %tiled_again = transform.structured.tile_using_for %tiled tile_sizes [] : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %i_again = transform.structured.tile_using_for %i tile_sizes [] : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %j_again = transform.structured.tile_using_for %j tile_sizes [] : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %k_again = transform.structured.tile_using_for %k tile_sizes [] : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.yield
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

// All-zero and empty size lists leave a single matmul with no loops. Reuse the
// first returned handle for the second tiling. The live tensor result makes
// erasing without cloning fail.
module attributes {transform.with_named_sequence} {
  func.func @zero(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %D = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %D : tensor<4x4xf32>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %zero = transform.structured.tile_using_for %target tile_sizes [0, 0, 0] : (!transform.any_op) -> !transform.any_op
    %empty = transform.structured.tile_using_for %zero tile_sizes [] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK:       builtin.module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @zero(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:      %D = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:      func.return %D : tensor<4x4xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %target = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %zero = transform.structured.tile_using_for %target tile_sizes [0, 0, 0] : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %empty = transform.structured.tile_using_for %zero tile_sizes [] : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.yield
// CHECK-NEXT:    }
// CHECK-NEXT:  }
