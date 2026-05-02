// RUN: xdsl-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

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

// acc.kernels uses `SingleBlockImplicitTerminator(TerminatorOp)`: a multi-
// block region fails the single-block check before terminator handling can
// run, so this case still exercises that trait branch.
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

// Generic-form region with a single, empty block — the implicit-terminator
// trait expects at least a terminator. (Pretty form auto-inserts the
// terminator; this generic spelling bypasses that helper.)
func.func @kernels_empty_block() {
  "acc.kernels"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
  ^bb0:
  }) : () -> ()
  func.return
}
// CHECK: Operation acc.kernels contains empty block in single-block region that expects at least a terminator

// -----

// Final op in the body is not a terminator at all — caught by the generic
// "must end in a terminator" check.
func.func @kernels_wrong_terminator(%arg0: i32) {
  "acc.kernels"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
    %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i32) -> i32
  }) : () -> ()
  func.return
}
// CHECK: Operation builtin.unrealized_conversion_cast terminates block in single-block region but is not a terminator

// -----

// Final op IS a terminator (`test.termop` carries `IsTerminator()`) but is
// not the specific terminator type required — directly exercises
// `SingleBlockImplicitTerminator(TerminatorOp)`'s type-mismatch branch and
// proves the trait was wired against `acc.terminator` specifically.
func.func @kernels_wrong_terminator_type() {
  "acc.kernels"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
    "test.termop"() : () -> ()
  }) : () -> ()
  func.return
}
// CHECK: 'acc.kernels' terminates with operation test.termop instead of acc.terminator

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
// trait is on `acc.firstprivate.recipe` and `acc.reduction.recipe`; one
// case is enough since it fires from the trait's verifier rather than
// from any per-op `verify_`.
func.func @recipe_leaks_outer_value(%outer: i32) {
  acc.private.recipe @p : i32 init {
  ^bb0(%a: i32):
    %r = arith.addi %outer, %a : i32
    acc.yield %r : i32
  }
  func.return
}
// CHECK: Operation using value defined out of its IsolatedFromAbove parent

// -----

// acc.reduction.recipe: empty init region is rejected (mirrors the same
// `verifyInitLikeSingleArgRegion` port used by the privatization recipes).
"acc.reduction.recipe"() <{sym_name = "r", type = i32, reductionOperator = #acc.reduction_operator<add>}> ({
}, {
}, {
}) : () -> ()
// CHECK: Operation does not verify: expects non-empty init region

// -----

// acc.reduction.recipe: empty combiner region is rejected.
"acc.reduction.recipe"() <{sym_name = "r", type = i32, reductionOperator = #acc.reduction_operator<add>}> ({
^bb0(%a: i32):
  "acc.yield"(%a) : (i32) -> ()
}, {
}, {
}) : () -> ()
// CHECK: Operation does not verify: expects non-empty combiner region

// -----

// acc.reduction.recipe: combiner region's first two args must match `type`.
"acc.reduction.recipe"() <{sym_name = "r", type = i32, reductionOperator = #acc.reduction_operator<add>}> ({
^bb0(%a: i32):
  "acc.yield"(%a) : (i32) -> ()
}, {
^bb0(%a: i64, %b: i64):
  "acc.yield"(%a) : (i64) -> ()
}, {
}) : () -> ()
// CHECK: Operation does not verify: expects combiner region with the first two arguments of the reduction type

// -----

// acc.reduction.recipe: every `acc.yield` in the combiner region must
// yield exactly one value of the reduction type. Distinct from the
// init-region check — combiner-region `acc.yield`s are walked
// post-block-shape verification.
"acc.reduction.recipe"() <{sym_name = "r", type = i32, reductionOperator = #acc.reduction_operator<add>}> ({
^bb0(%a: i32):
  "acc.yield"(%a) : (i32) -> ()
}, {
^bb0(%a: i32, %b: i32):
  %c0 = "arith.constant"() <{value = 0 : i64}> : () -> i64
  "acc.yield"(%c0) : (i64) -> ()
}, {
}) : () -> ()
// CHECK: Operation does not verify: expects combiner region to yield a value of the reduction type

// -----

// `DataEntryOilist` rejects each of its four clauses appearing twice on
// parse. Cover all four keywords — the duplicate-detection branch is per-
// keyword, so a single shared case wouldn't prove each branch fires.
func.func @copyin_duplicate_var_ptr_ptr(%a : memref<10xf32>, %p : memref<memref<10xf32>>) {
  %r = acc.copyin varPtr(%a : memref<10xf32>) varPtrPtr(%p : memref<memref<10xf32>>) varPtrPtr(%p : memref<memref<10xf32>>) -> memref<10xf32>
  func.return
}
// CHECK: 'varPtrPtr' clause specified twice

// -----

func.func @copyin_duplicate_bounds(%a : memref<10xf32>, %c0 : index, %c9 : index) {
  %b = acc.bounds lowerbound(%c0 : index) upperbound(%c9 : index)
  %r = acc.copyin varPtr(%a : memref<10xf32>) bounds(%b) bounds(%b) -> memref<10xf32>
  func.return
}
// CHECK: 'bounds' clause specified twice

// -----

func.func @copyin_duplicate_async(%a : memref<10xf32>) {
  %r = acc.copyin varPtr(%a : memref<10xf32>) async async -> memref<10xf32>
  func.return
}
// CHECK: 'async' clause specified twice

// -----

func.func @copyin_duplicate_recipe(%a : memref<10xf32>) {
  %r = acc.copyin varPtr(%a : memref<10xf32>) recipe(@r1) recipe(@r2) -> memref<10xf32>
  func.return
}
// CHECK: 'recipe' clause specified twice

// -----

// acc.terminator's HasParent constraint accepts only the OpenACC region ops
// that lack a yield (kernels / data / host_data); using it directly under a
// `func.func` must fail the parent check.
func.func @terminator_wrong_parent() {
  "acc.terminator"() : () -> ()
}
// CHECK: 'acc.terminator' expects parent op to be one of 'acc.kernels', 'acc.data', 'acc.host_data'

// -----

// acc.data: 2.6.5 mandates at least one operand or `defaultAttr`. An empty
// op with no operands and no default attribute fails the per-op verifier.
func.func @data_empty_no_default() {
  acc.data {
  }
  func.return
}
// CHECK: at least one operand or the default attribute must appear on the data operation

// -----

// acc.data: each `dataOperands` value must be defined by one of the OpenACC
// data-clause ops (copyin / create / present / …). A bare memref operand
// fails the per-op verifier.
func.func @data_wrong_defining_op(%a : memref<10xf32>) {
  acc.data dataOperands(%a : memref<10xf32>) {
  }
  func.return
}
// CHECK: expect data entry/exit operation or acc.getdeviceptr as defining op

// -----

// acc.host_data: at least one operand is required, period (no `default`
// escape hatch like acc.data has).
func.func @host_data_empty() {
  acc.host_data {
  }
  func.return
}
// CHECK: at least one operand must appear on the host_data operation

// -----

// acc.host_data: every operand must be defined by acc.use_device. Other
// data-clause ops (e.g. acc.copyin) are rejected.
func.func @host_data_wrong_defining_op(%a : memref<10xf32>) {
  %0 = acc.copyin varPtr(%a : memref<10xf32>) -> memref<10xf32>
  acc.host_data dataOperands(%0 : memref<10xf32>) {
  }
  func.return
}
// CHECK: expect data entry operation as defining op

// -----

// acc.enter_data: 2.6.6 requires at least one copyin/create/attach clause.
// An op with empty `dataOperands` fails the per-op verifier.
func.func @enter_data_empty() {
  acc.enter_data
  func.return
}
// CHECK: at least one operand must be present in dataOperands on the enter data operation

// -----

// acc.enter_data: each `dataOperands` value must be defined by an
// `acc.copyin` / `acc.create` / `acc.attach`. A bare memref fails.
func.func @enter_data_wrong_defining_op(%a : memref<10xf32>) {
  acc.enter_data dataOperands(%a : memref<10xf32>)
  func.return
}
// CHECK: expect data entry operation as defining op

// -----

// acc.enter_data: the bare `async` UnitAttr and an `asyncOperand` cannot
// both be set — the keyword-only spelling is mutually exclusive with the
// operand-bearing spelling.
func.func @enter_data_async_conflict(%a : memref<10xf32>, %v : i64) {
  %d = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
  "acc.enter_data"(%v, %d) <{async, operandSegmentSizes = array<i32: 0, 1, 0, 0, 1>}> : (i64, memref<10xf32>) -> ()
  func.return
}
// CHECK: async attribute cannot appear with asyncOperand

// -----

// acc.enter_data: the bare `wait` UnitAttr and `waitOperands` cannot both
// be set.
func.func @enter_data_wait_conflict(%a : memref<10xf32>, %w : i32) {
  %d = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
  "acc.enter_data"(%w, %d) <{wait, operandSegmentSizes = array<i32: 0, 0, 0, 1, 1>}> : (i32, memref<10xf32>) -> ()
  func.return
}
// CHECK: wait attribute cannot appear with waitOperands

// -----

// acc.enter_data: `wait_devnum` requires `waitOperands` to be non-empty.
func.func @enter_data_wait_devnum_alone(%a : memref<10xf32>, %dn : i64) {
  %d = acc.create varPtr(%a : memref<10xf32>) -> memref<10xf32>
  "acc.enter_data"(%dn, %d) <{operandSegmentSizes = array<i32: 0, 0, 1, 0, 1>}> : (i64, memref<10xf32>) -> ()
  func.return
}
// CHECK: wait_devnum cannot appear without waitOperands

// -----

// acc.exit_data: 2.6.6 requires at least one copyout/delete/detach clause.
func.func @exit_data_empty() {
  acc.exit_data
  func.return
}
// CHECK: at least one operand must be present in dataOperands on the exit data operation

// -----

// acc.exit_data: bare `async` UnitAttr and an `asyncOperand` are mutually
// exclusive.
func.func @exit_data_async_conflict(%a : memref<10xf32>, %v : i64) {
  %d = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
  "acc.exit_data"(%v, %d) <{async, operandSegmentSizes = array<i32: 0, 1, 0, 0, 1>}> : (i64, memref<10xf32>) -> ()
  func.return
}
// CHECK: async attribute cannot appear with asyncOperand

// -----

// acc.exit_data: bare `wait` UnitAttr and `waitOperands` are mutually
// exclusive.
func.func @exit_data_wait_conflict(%a : memref<10xf32>, %w : i32) {
  %d = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
  "acc.exit_data"(%w, %d) <{wait, operandSegmentSizes = array<i32: 0, 0, 0, 1, 1>}> : (i32, memref<10xf32>) -> ()
  func.return
}
// CHECK: wait attribute cannot appear with waitOperands

// -----

// acc.exit_data: `wait_devnum` requires `waitOperands` to be non-empty.
func.func @exit_data_wait_devnum_alone(%a : memref<10xf32>, %dn : i64) {
  %d = acc.getdeviceptr varPtr(%a : memref<10xf32>) -> memref<10xf32>
  "acc.exit_data"(%dn, %d) <{operandSegmentSizes = array<i32: 0, 0, 1, 0, 1>}> : (i64, memref<10xf32>) -> ()
  func.return
}
// CHECK: wait_devnum cannot appear without waitOperands

// -----

// acc.update: empty `dataOperands` fails the per-op verifier.
func.func @update_empty() {
  acc.update
  func.return
}
// CHECK: at least one value must be present in dataOperands

// -----

// acc.update: every `dataOperands` value must be defined by an
// `acc.update_device` / `acc.update_host` / `acc.getdeviceptr`. A bare
// memref fails.
func.func @update_wrong_defining_op(%a : memref<f32>) {
  acc.update dataOperands(%a : memref<f32>)
  func.return
}
// CHECK: expect data entry/exit operation or acc.getdeviceptr as defining op
