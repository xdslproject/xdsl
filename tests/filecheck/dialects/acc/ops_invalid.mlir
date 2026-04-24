// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

func.func @empty_block() {
  "acc.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
  ^bb0:
  }) : () -> ()
  func.return
}
// CHECK: acc.parallel

// -----

func.func @wrong_terminator(%arg0: i32) {
  "acc.parallel"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
    %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i32) -> i32
  }) : () -> ()
  func.return
}
// CHECK: builtin.unrealized_conversion_cast

// -----

builtin.module {
  "acc.yield"() : () -> ()
}
// CHECK: 'acc.yield' expects parent op 'acc.parallel'
