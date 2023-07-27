// RUN: xdsl-opt %s

// This file tests that the parser does not assert in case the block
// ordering seemingly indicates that an operand is not yet defined.

builtin.module {
  func.func @bb_ordered_by_dominance() {
    "test.termop"() [^1] : () -> ()

  ^1:
    %1 = arith.constant 1
    "test.termop"() [^2] : () -> ()

  ^2:
    %2 = arith.addi %1, %1 : i64
    "test.termop"() [^1] : () -> ()
  }

  func.func @bb_ordered_against_dominance() {
    "test.termop"() [^1] : () -> ()

  ^2:
    // Previously, this operation did not parse due to %1 seemingly
    // not yet being defined.
    %2 = arith.addi %1, %1 : i64
    "test.termop"() [^1] : () -> ()

  ^1:
    %1 = arith.constant 1
    "test.termop"() [^2] : () -> ()
  }
}
