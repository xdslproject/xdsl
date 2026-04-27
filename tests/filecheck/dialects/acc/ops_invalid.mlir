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
// CHECK: 'acc.yield' expects parent op

// -----

func.func @serial_empty_block() {
  "acc.serial"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0>}> ({
  ^bb0:
  }) : () -> ()
  func.return
}
// CHECK: Operation acc.serial contains empty block in single-block region that expects at least a terminator

// -----

func.func @serial_wrong_terminator(%arg0: i32) {
  "acc.serial"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0>}> ({
    %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i32) -> i32
  }) : () -> ()
  func.return
}
// CHECK: Operation builtin.unrealized_conversion_cast terminates block in single-block region but is not a terminator

// -----

// acc.kernels uses upstream's NoTerminator body modeling (AnyRegion with no
// implicit terminator), so empty blocks and non-terminator final ops are
// accepted; the only verifier failure unique to that trait is a multi-block
// region.
func.func @kernels_multi_block() {
  "acc.kernels"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
  ^bb0:
    "cf.br"()[^bb1] : () -> ()
  ^bb1:
  }) : () -> ()
  func.return
}
// CHECK: 'acc.kernels' does not contain single-block regions

// -----

// acc.bounds requires either `extent` or `upperbound` (or both) to be set.
func.func @bounds_missing_extent_and_upperbound() {
  %b = "acc.bounds"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0>}> : () -> !acc.data_bounds_ty
  func.return
}
// CHECK: Operation does not verify: expected extent or upperbound.
