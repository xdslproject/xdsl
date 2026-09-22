// RUN: xdsl-opt %s -p transform-interpreter --split-input-file --verify-diagnostics | filecheck %s

// Reject targets that are not structured linalg operations.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %tiled, %loop = transform.structured.tile_using_for %root tile_sizes [2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.tile_using_for supports only structured linalg targets{{$}}

// -----

// Reject dynamic tile sizes, even for an empty target handle.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %empty = transform.structured.match ops{[]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %empty tile_sizes [%root] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.tile_using_for does not yet support dynamic tile sizes{{$}}

// -----

// Reject scalable tile sizes, even for an empty target handle.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %empty = transform.structured.match ops{[]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %empty tile_sizes [[2]] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.tile_using_for does not yet support scalable tile sizes{{$}}

// -----

// Reject interchange, even for an empty target handle.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %empty = transform.structured.match ops{[]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %empty tile_sizes [2] interchange = [0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.tile_using_for does not yet support interchange{{$}}

// -----

// Reject negative tile sizes, even for an empty target handle.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %empty = transform.structured.match ops{[]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %empty tile_sizes [-1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.tile_using_for requires nonnegative tile sizes{{$}}

// -----

// Do not silently truncate tile sizes beyond the target loop rank.
module attributes {transform.with_named_sequence} {
  func.func @matmul(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %D = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %D : tensor<4x4xf32>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:4 = transform.structured.tile_using_for %target tile_sizes [2, 2, 2, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.tile_using_for expected at most 3 tile sizes for linalg.matmul, got 4{{$}}

// -----

// Validate all target types before tiling any of them.
module attributes {transform.with_named_sequence} {
  func.func @matmul(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %D = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %D : tensor<4x4xf32>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %targets = transform.structured.match ops{["linalg.matmul", "func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %targets tile_sizes [2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: {{^}}transform.structured.tile_using_for supports only structured linalg targets{{$}}
// CHECK-NOT: "scf.for"
