// RUN: xdsl-opt -p transform-interpreter %s | filecheck %s

// Tile a tensor matmul through the transform interpreter, using MLIR-compatible
// schedule syntax. Check all three loops and the slices and yields that carry
// the accumulator through the reduction. The schedule remains in the output.

module attributes {transform.with_named_sequence} {
  func.func @matmul(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>) outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    // One result for the tiled op, plus one loop per non-zero tile size: 1 + 3 = 4.
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul tile_sizes [32, 32, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK:       builtin.module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @matmul(%A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
// CHECK-NEXT:      %0 = arith.constant 0 : index
// CHECK-NEXT:      %1 = arith.constant 128 : index
// CHECK-NEXT:      %2 = arith.constant 128 : index
// CHECK-NEXT:      %3 = arith.constant 128 : index
// CHECK-NEXT:      %4 = arith.constant 32 : index
// CHECK-NEXT:      %5 = arith.constant 32 : index
// CHECK-NEXT:      %6 = arith.constant 32 : index
// CHECK-NEXT:      %7 = scf.for %8 = %0 to %1 step %4 iter_args(%9 = %C) -> (tensor<128x128xf32>) {
// CHECK-NEXT:        %10 = scf.for %11 = %0 to %2 step %5 iter_args(%12 = %9) -> (tensor<128x128xf32>) {
// CHECK-NEXT:          %13 = scf.for %14 = %0 to %3 step %6 iter_args(%15 = %12) -> (tensor<128x128xf32>) {
// CHECK-NEXT:            %16 = tensor.extract_slice %A[%8, %14] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
// CHECK-NEXT:            %17 = tensor.extract_slice %B[%14, %11] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
// CHECK-NEXT:            %18 = tensor.extract_slice %15[%8, %11] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
// CHECK-NEXT:            %19 = linalg.matmul ins(%16, %17 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK-NEXT:            %20 = tensor.insert_slice %19 into %15[%8, %11] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<128x128xf32>
// CHECK-NEXT:            scf.yield %20 : tensor<128x128xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %13 : tensor<128x128xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %10 : tensor<128x128xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %7 : tensor<128x128xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %matmul = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled, %loops, %loops_1, %loops_2 = transform.structured.tile_using_for %matmul tile_sizes [32, 32, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.yield
// CHECK-NEXT:    }
// CHECK-NEXT:  }
