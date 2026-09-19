// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

%to_match = "test.op"() : () -> !transform.any_op
// CHECK: interface must be one of LinalgOp, TilingInterface, LoopLikeInterface (0 to 2), got 7
%matched = "transform.structured.match"(%to_match) <{interface = 7 : i32}> : (!transform.any_op) -> !transform.any_op

// -----

%to_match = "test.op"() : () -> !transform.any_op
// CHECK: Expected attribute i32 but got i64
%matched = "transform.structured.match"(%to_match) <{interface = 1 : i64}> : (!transform.any_op) -> !transform.any_op
