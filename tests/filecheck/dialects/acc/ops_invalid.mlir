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

// -----

// acc.private.recipe: empty init region is rejected (mirrors upstream's
// `verifyInitLikeSingleArgRegion`).
"acc.private.recipe"() <{sym_name = "p", type = i32}> ({
}, {
}) : () -> ()
// CHECK: Operation does not verify: expects non-empty init region

// -----

// acc.private.recipe: init region's first arg type must match `type`.
"acc.private.recipe"() <{sym_name = "p", type = i32}> ({
^bb0(%a: i64):
  "acc.yield"() : () -> ()
}, {
}) : () -> ()
// CHECK: Operation does not verify: expects init region first argument of the privatization type

// -----

// acc.private.recipe: when `destroy` is supplied, its first arg must
// match `type` too.
"acc.private.recipe"() <{sym_name = "p", type = i32}> ({
^bb0(%a: i32):
  "acc.yield"() : () -> ()
}, {
^bb0(%a: i64):
  "acc.yield"() : () -> ()
}) : () -> ()
// CHECK: Operation does not verify: expects destroy region first argument of the privatization type

// -----

// acc.firstprivate.recipe: empty copy region is rejected.
"acc.firstprivate.recipe"() <{sym_name = "p", type = i32}> ({
^bb0(%a: i32):
  "acc.yield"() : () -> ()
}, {
}, {
}) : () -> ()
// CHECK: Operation does not verify: expects non-empty copy region

// -----

// acc.firstprivate.recipe: copy region must take two args of `type`.
"acc.firstprivate.recipe"() <{sym_name = "p", type = i32}> ({
^bb0(%a: i32):
  "acc.yield"() : () -> ()
}, {
^bb0(%a: i32):
  "acc.yield"() : () -> ()
}, {
}) : () -> ()
// CHECK: Operation does not verify: expects copy region with two arguments of the privatization type

// -----

// `IsolatedFromAbove` on `acc.private.recipe` is enforced by the verifier:
// nested ops cannot reference SSA values defined outside the recipe. Same
// trait is on `acc.firstprivate.recipe`; one case is enough since it
// fires from the trait's verifier rather than the per-op `verify_`.
func.func @recipe_leaks_outer_value(%outer: i32) {
  acc.private.recipe @p : i32 init {
  ^bb0(%a: i32):
    %r = arith.addi %outer, %a : i32
    acc.yield %r : i32
  }
  func.return
}
// CHECK: Operation using value defined out of its IsolatedFromAbove parent
