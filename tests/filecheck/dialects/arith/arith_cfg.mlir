// RUN: xdsl-opt %s

// This file tests that the parser does not assert in case the block
// ordering seemingly indicates that an operand is not yet defined.

builtin.module {
  func.func @bb_ordered_by_dominance(%0 : i1) {
    "cf.br"(%0) [^1] : (i1) -> ()

  ^1:
    %1 = arith.constant 1
    "cf.br"(%0) [^2] : (i1) -> ()

  ^2:
    %2 = arith.addi %1, %1 : i64
    "cf.br"(%0) [^1] : (i1) -> ()
  }

  func.func @bb_ordered_against_dominance(%0 : i1) {
    "cf.br"(%0) [^1] : (i1) -> ()

  ^2:
    // Previously, this operation did not parse due to %1 seemingly
    // not yet being defined.
    %2 = arith.addi %1, %1 : i64
    "cf.br"(%0) [^1] : (i1) -> ()

  ^1:
    %1 = arith.constant 1
    "cf.br"(%0) [^2] : (i1) -> ()
  }
}
