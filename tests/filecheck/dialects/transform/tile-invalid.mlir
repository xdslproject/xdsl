// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

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
