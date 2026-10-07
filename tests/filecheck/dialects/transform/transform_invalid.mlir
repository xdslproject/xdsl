// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

%to_match = "test.op"() : () -> !transform.any_op
// CHECK: interface must be one of LinalgOp, TilingInterface, LoopLikeInterface (0 to 2), got 7
%matched = "transform.structured.match"(%to_match) <{interface = 7 : i32}> : (!transform.any_op) -> !transform.any_op

// -----

%to_match = "test.op"() : () -> !transform.any_op
// CHECK: Expected attribute i32 but got i64
%matched = "transform.structured.match"(%to_match) <{interface = 1 : i64}> : (!transform.any_op) -> !transform.any_op

// -----

%to_match = "test.op"() : () -> !transform.any_op
// CHECK: Expected `LinalgOp`, `TilingInterface`, or `LoopLikeInterface`.
%matched = transform.structured.match interface{Nope} in %to_match : (!transform.any_op) -> !transform.any_op

// -----

// Tile

// The number of loop results must match the nonzero tile sizes.
%target = "test.op"() : () -> !transform.any_op
%result:1 = "transform.structured.tile_using_for"(%target) <{static_sizes = array<i64: 2, 0>}> : (!transform.any_op) -> !transform.any_op

// CHECK: expected 1 loop results for nonzero tile sizes, got 0

// -----

// Zero tile sizes do not produce loop results.
%target = "test.op"() : () -> !transform.any_op
%result:2 = "transform.structured.tile_using_for"(%target) <{static_sizes = array<i64: 0, 0>}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

// CHECK: expected 0 loop results for nonzero tile sizes, got 1

// -----

// Each dynamic sentinel needs a dynamic size operand.
%target = "test.op"() : () -> !transform.any_op
%result:2 = "transform.structured.tile_using_for"(%target) <{static_sizes = array<i64: -9223372036854775808>}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

// CHECK: expected 1 dynamic size operands, got 0

// -----

// Static sizes must not have extra dynamic operands.
%target = "test.op"() : () -> !transform.any_op
%result:2 = "transform.structured.tile_using_for"(%target, %target) <{static_sizes = array<i64: 2>}> : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

// CHECK: expected 0 dynamic size operands, got 1

// -----

// Explicit scalable flags must have the same length as the sizes.
%target = "test.op"() : () -> !transform.any_op
%result:2 = "transform.structured.tile_using_for"(%target) <{static_sizes = array<i64: 2, 0>, scalable_sizes = array<i1: false>}> : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

// CHECK: expected 2 scalable size flags, got 1

// -----

// An absent size list is empty, so it cannot have scalable flags.
%target = "test.op"() : () -> !transform.any_op
%result:1 = "transform.structured.tile_using_for"(%target) <{scalable_sizes = array<i1: false>}> : (!transform.any_op) -> !transform.any_op

// CHECK: expected 0 scalable size flags, got 1

// -----

// Yield

transform.named_sequence @wrong_type(%arg: !transform.any_op) -> !transform.any_value {
  transform.yield %arg : !transform.any_op
}

// CHECK: Expected yielded values to have the same types as the named sequence output types
