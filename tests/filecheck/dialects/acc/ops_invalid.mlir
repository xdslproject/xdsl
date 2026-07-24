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

// -----

// acc.declare_enter: empty `dataOperands` fails the per-op verifier
// (mirrors upstream's `checkDeclareOperands(requireAtLeastOneOperand=true)`).
func.func @declare_enter_empty() {
  %t = acc.declare_enter
  acc.declare_exit token(%t)
  func.return
}
// CHECK: at least one operand must appear on the declare operation

// -----

// acc.declare_enter: every `dataOperands` value must be defined by one
// of the eight allowed data-clause ops (per `checkDeclareOperands`). A
// bare block argument fails.
func.func @declare_enter_wrong_defining_op(%a : memref<f32>) {
  %t = acc.declare_enter dataOperands(%a : memref<f32>)
  acc.declare_exit token(%t) dataOperands(%a : memref<f32>)
  func.return
}
// CHECK: expect valid declare data entry operation or acc.getdeviceptr as defining op

// -----

// acc.declare_exit without a `token`: empty `dataOperands` fails. The
// `token`-bearing form would relax this — covered by the positive
// roundtrip `declare_exit_token_only` in `ops.mlir`.
func.func @declare_exit_empty_no_token() {
  acc.declare_exit
  func.return
}
// CHECK: at least one operand must appear on the declare operation

// -----

// acc.declare: empty `dataOperands` fails (the structured declare always
// requires at least one data operand to scope the implicit data region).
func.func @declare_empty() {
  acc.declare {
  }
  func.return
}
// CHECK: at least one operand must appear on the declare operation

// -----

// acc.declare: every `dataOperands` value must be defined by one of the
// eight allowed data-clause ops; a bare block-argument memref fails.
func.func @declare_wrong_defining_op(%a : memref<f32>) {
  acc.declare dataOperands(%a : memref<f32>) {
  }
  func.return
}
// CHECK: expect valid declare data entry operation or acc.getdeviceptr as defining op

// -----

// acc.loop: at least one of auto / independent / seq must apply to the
// device-`none` (default) device type. A bare loop with no parallelism
// markers fails this check.
func.func @loop_missing_par_mode() {
  acc.loop {
    acc.yield
  }
  func.return
}
// CHECK: at least one of auto, independent, seq must be present

// -----

// acc.loop: gang / worker / vector cannot coexist with seq for the same
// device type — the verifier catches `seq + gang` on `#none`. The bare
// `gang` keyword writes to the `gang` property directly via GangClause,
// matching how a real source emits it.
func.func @loop_seq_with_gang() {
  acc.loop gang {
    acc.yield
  } attributes {seq = [#acc.device_type<none>]}
  func.return
}
// CHECK: gang, worker or vector cannot appear with seq

// -----

// acc.loop: the same device type cannot appear twice across auto /
// independent / seq.
func.func @loop_auto_seq_same_dt() {
  acc.loop {
    acc.yield
  } attributes {auto_ = [#acc.device_type<none>], seq = [#acc.device_type<none>]}
  func.return
}
// CHECK: only one of auto, independent, seq can be present at the same time

// -----

// acc.loop: duplicate device type in the gang attribute fires a verifier
// error (mirrors upstream's `checkDeviceTypes` helper). The
// `gang([#dt, #dt])` keyword-only DT spelling writes the array directly
// to the `gang` property via GangClause.
func.func @loop_duplicate_gang_dt() {
  acc.loop gang([#acc.device_type<none>, #acc.device_type<none>]) {
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  func.return
}
// CHECK: duplicate device_type `none` found in gang attribute

// -----

// acc.loop: an `unstructured` loop carries no induction variables — pairing
// it with a `control(...)` clause is rejected.
func.func @loop_unstructured_with_control(%lb : index, %ub : index, %st : index) {
  acc.loop control(%iv : index) = (%lb : index) to (%ub : index) step (%st : index) {
    acc.yield
  } attributes {independent = [#acc.device_type<none>], unstructured}
  func.return
}
// CHECK: unstructured acc.loop must not have induction variables

// -----

// acc.loop: lowerbound / upperbound / step counts must match each other.
// Generic-form input is the only way to land mismatched counts; the
// declarative parser only accepts equal-length lists.
func.func @loop_unequal_counts(%lb : index, %ub : index, %st : index) {
  "acc.loop"(%lb, %ub) <{independent = [#acc.device_type<none>], operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
  ^bb0(%iv: index):
    "acc.yield"() : () -> ()
  }) : (index, index) -> ()
  func.return
}
// CHECK: number of upperbounds expected to be the same as number of steps

// -----

// acc.loop: lowerbound count must match upperbound count. Generic-form
// input is the only way to reach this check (the declarative parser uses
// the same induction-var count for all three lists). Pick segment sizes
// where upperbound == step but lowerbound differs, so the upperbound/step
// check passes and the lowerbound check fires.
func.func @loop_lb_ub_mismatch(%lb : index, %ub : index, %st : index) {
  "acc.loop"(%lb, %ub, %ub, %st, %st) <{independent = [#acc.device_type<none>], operandSegmentSizes = array<i32: 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
  ^bb0(%iv: index, %iv2: index):
    "acc.yield"() : () -> ()
  }) : (index, index, index, index, index) -> ()
  func.return
}
// CHECK: number of upperbounds expected to be the same as number of lowerbounds

// -----

// acc.loop: inclusiveUpperbound array size must match upperbound count.
func.func @loop_inclusive_size_mismatch(%lb : index, %ub : index, %st : index) {
  "acc.loop"(%lb, %ub, %st) <{independent = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true, false>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0>}> ({
  ^bb0(%iv: index):
    "acc.yield"() : () -> ()
  }) : (index, index, index) -> ()
  func.return
}
// CHECK: inclusiveUpperbound size is expected to be the same as upperbound size

// -----

// acc.loop: a `collapse` attribute requires `collapseDeviceType` too.
func.func @loop_collapse_no_dt() {
  acc.loop {
    acc.yield
  } attributes {collapse = [1 : i64], independent = [#acc.device_type<none>]}
  func.return
}
// CHECK: collapse device_type attr must be define when collapse attr is present

// -----

// acc.loop: collapse and collapseDeviceType arrays must have matching counts.
func.func @loop_collapse_count_mismatch() {
  acc.loop {
    acc.yield
  } attributes {collapse = [1 : i64, 2 : i64], collapseDeviceType = [#acc.device_type<none>], independent = [#acc.device_type<none>]}
  func.return
}
// CHECK: collapse attribute count must match collapse device_type count

// -----

// acc.loop: gang_operands present without `gangOperandsArgType` fails.
func.func @loop_gang_no_arg_type(%v : i64) {
  "acc.loop"(%v) <{independent = [#acc.device_type<none>], operandSegmentSizes = array<i32: 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0>}> ({
    "acc.yield"() : () -> ()
  }) : (i64) -> ()
  func.return
}
// CHECK: gangOperandsArgType attribute must be defined when gang operands are present

// -----

// acc.loop: gangOperandsArgType count must match gang_operands count.
func.func @loop_gang_arg_type_mismatch(%v : i64) {
  "acc.loop"(%v, %v) <{independent = [#acc.device_type<none>], gangOperandsArgType = [#acc.gang_arg_type<Num>], gangOperandsDeviceType = [#acc.device_type<none>], gangOperandsSegments = array<i32: 2>, operandSegmentSizes = array<i32: 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0>}> ({
    "acc.yield"() : () -> ()
  }) : (i64, i64) -> ()
  func.return
}
// CHECK: gangOperandsArgType attribute count must match gangOperands count

// -----

// acc.loop: worker_num_operands count must match worker_num_operands_device_type
// count. Generic-form bypasses the declarative parser, which would always
// populate one DT entry per operand.
func.func @loop_worker_dt_mismatch(%v : i64) {
  "acc.loop"(%v, %v) <{independent = [#acc.device_type<none>], workerNumOperandsDeviceType = [#acc.device_type<none>], operandSegmentSizes = array<i32: 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0>}> ({
    "acc.yield"() : () -> ()
  }) : (i64, i64) -> ()
  func.return
}
// CHECK: worker operands count must match worker device_type count

// -----

// acc.loop: vector_operands count must match vector_operands_device_type
// count.
func.func @loop_vector_dt_mismatch(%v : i64) {
  "acc.loop"(%v, %v) <{independent = [#acc.device_type<none>], vectorOperandsDeviceType = [#acc.device_type<none>], operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>}> ({
    "acc.yield"() : () -> ()
  }) : (i64, i64) -> ()
  func.return
}
// CHECK: vector operands count must match vector device_type count

// -----

// acc.loop: tile operand count must match the sum of tileOperandsSegments.
func.func @loop_tile_segment_count_mismatch(%v : i64) {
  "acc.loop"(%v, %v) <{independent = [#acc.device_type<none>], tileOperandsDeviceType = [#acc.device_type<none>], tileOperandsSegments = array<i32: 1>, operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>}> ({
    "acc.yield"() : () -> ()
  }) : (i64, i64) -> ()
  func.return
}
// CHECK: tile operand count does not match count in segments

// -----

// acc.loop: tileOperandsSegments count must match tileOperandsDeviceType count.
func.func @loop_tile_segment_dt_mismatch(%v : i64) {
  "acc.loop"(%v, %v) <{independent = [#acc.device_type<none>], tileOperandsDeviceType = [#acc.device_type<none>, #acc.device_type<nvidia>], tileOperandsSegments = array<i32: 2>, operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0>}> ({
    "acc.yield"() : () -> ()
  }) : (i64, i64) -> ()
  func.return
}
// CHECK: tile segment count does not match device_type count

// -----

// acc.loop: parser rejects `combined(...)` with an unknown construct keyword.
func.func @loop_combined_bad_keyword(%lb : index, %ub : index, %st : index) {
  acc.loop combined(banana) control(%iv : index) = (%lb : index) to (%ub : index) step (%st : index) {
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  func.return
}
// CHECK: expected compute construct name

// -----

// acc.loop: gang clause with empty group `{}` is rejected.
func.func @loop_gang_empty_group(%lb : index, %ub : index, %st : index) {
  acc.loop gang({}) control(%iv : index) = (%lb : index) to (%ub : index) step (%st : index) {
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  func.return
}
// CHECK: expect at least one of num, dim or static values

// -----

// acc.loop: trailing comma inside a gang group fires the parser's
// "new value expected after comma" error.
func.func @loop_gang_trailing_comma(%v : i64, %lb : index, %ub : index, %st : index) {
  acc.loop gang({num=%v : i64,}) control(%iv : index) = (%lb : index) to (%ub : index) step (%st : index) {
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  func.return
}
// CHECK: new value expected after comma

// -----

// acc.loop: `control(...)` with too few lower-bound operands raises a
// parse error. The declarative parser uses the induction-var count to
// bound the operand list.
func.func @loop_control_too_few_lb(%lb : index, %ub : index, %st : index) {
  acc.loop control(%iv : index, %iv2 : index) = (%lb : index) to (%ub, %ub : index, index) step (%st, %st : index, index) {
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  func.return
}
// CHECK: expected 2 operands

// -----

// acc.loop: `control(...)` with too few lower-bound types raises a parse
// error.
func.func @loop_control_too_few_lb_types(%lb : index, %ub : index, %st : index) {
  acc.loop control(%iv : index, %iv2 : index) = (%lb, %lb : index) to (%ub, %ub : index, index) step (%st, %st : index, index) {
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  func.return
}
// CHECK: expected 2 types

// -----

// acc.loop: duplicate device_type in `worker` attribute fires the
// duplicate-DT verifier. `worker([#dt, #dt])` is the keyword-only DT
// spelling that writes the array directly to the `worker` property.
func.func @loop_duplicate_worker_dt() {
  acc.loop worker([#acc.device_type<none>, #acc.device_type<none>]) {
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  func.return
}
// CHECK: duplicate device_type `none` found in worker attribute

// -----

// acc.loop: duplicate device_type in `vector` attribute fires the
// duplicate-DT verifier.
func.func @loop_duplicate_vector_dt() {
  acc.loop vector([#acc.device_type<none>, #acc.device_type<none>]) {
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  func.return
}
// CHECK: duplicate device_type `none` found in vector attribute

// -----

// acc.loop: duplicate device_type in `workerNumOperandsDeviceType`
// (generic form — declarative parser would never produce duplicates).
func.func @loop_duplicate_worker_num_operands_dt(%v : i64) {
  "acc.loop"(%v, %v) <{independent = [#acc.device_type<none>], workerNumOperandsDeviceType = [#acc.device_type<none>, #acc.device_type<none>], operandSegmentSizes = array<i32: 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0>}> ({
    "acc.yield"() : () -> ()
  }) : (i64, i64) -> ()
  func.return
}
// CHECK: duplicate device_type `none` found in workerNumOperandsDeviceType attribute

// -----

// acc.loop: duplicate device_type in `vectorOperandsDeviceType`.
func.func @loop_duplicate_vector_operands_dt(%v : i64) {
  "acc.loop"(%v, %v) <{independent = [#acc.device_type<none>], vectorOperandsDeviceType = [#acc.device_type<none>, #acc.device_type<none>], operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>}> ({
    "acc.yield"() : () -> ()
  }) : (i64, i64) -> ()
  func.return
}
// CHECK: duplicate device_type `none` found in vectorOperandsDeviceType attribute

// -----

// acc.loop: duplicate device_type in `collapseDeviceType` attribute.
func.func @loop_duplicate_collapse_dt() {
  acc.loop {
    acc.yield
  } attributes {collapse = [1 : i64, 1 : i64], collapseDeviceType = [#acc.device_type<none>, #acc.device_type<none>], independent = [#acc.device_type<none>]}
  func.return
}
// CHECK: duplicate device_type `none` found in collapseDeviceType attribute

// -----

// acc.init nested in acc.parallel — runtime ops cannot be nested in a
// compute construct (parallel/serial/kernels/loop).
func.func @init_in_parallel() {
  acc.parallel {
    acc.init
    acc.yield
  }
  func.return
}
// CHECK: 'acc.init' op cannot be nested in a compute operation

// -----

// acc.init nested transitively (through acc.serial) — the parent walk
// must traverse all ancestors, not just the direct parent.
func.func @init_in_serial() {
  acc.serial {
    acc.init
    acc.yield
  }
  func.return
}
// CHECK: 'acc.init' op cannot be nested in a compute operation

// -----

// acc.shutdown nested in acc.kernels.
func.func @shutdown_in_kernels() {
  acc.kernels {
    acc.shutdown
    acc.terminator
  }
  func.return
}
// CHECK: 'acc.shutdown' op cannot be nested in a compute operation

// -----

// acc.set nested in acc.loop.
func.func @set_in_loop(%lb : index, %ub : index, %st : index) {
  acc.loop control(%iv : index) = (%lb : index) to (%ub : index) step (%st : index) {
    acc.set attributes {device_type = #acc.device_type<nvidia>}
    acc.yield
  } attributes {independent = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
  func.return
}
// CHECK: 'acc.set' op cannot be nested in a compute operation

// -----

// acc.set with no operands and no device_type — at least one of
// default_async/device_num/device_type must appear.
func.func @set_empty() {
  acc.set
  func.return
}
// CHECK: at least one default_async, device_num, or device_type operand must appear

// -----

// acc.wait: the bare `async` UnitAttr and an `asyncOperand` cannot both be
// set. The pretty form can only emit one or the other, so build the
// conflicting state via generic form.
func.func @wait_async_conflict(%v : i32) {
  "acc.wait"(%v) <{async, operandSegmentSizes = array<i32: 0, 1, 0, 0>}> : (i32) -> ()
  func.return
}
// CHECK: async attribute cannot appear with asyncOperand

// -----

// acc.wait: `wait_devnum` requires `waitOperands` to be non-empty.
func.func @wait_devnum_alone(%dn : i32) {
  acc.wait wait_devnum(%dn : i32)
  func.return
}
// CHECK: wait_devnum cannot appear without waitOperands

// -----

// acc.routine: at most one of `gang`/`worker`/`vector`/`seq` may be set
// for the `none` device type — built via generic form because the pretty
// form also goes through this same verifier.
func.func @routine_base_parallelism() {
  func.return
}
"acc.routine"() <{func_name = @routine_base_parallelism, sym_name = "rt_bad_base", gang = [#acc.device_type<none>], worker = [#acc.device_type<none>]}> : () -> ()
// CHECK: only one of `gang`, `worker`, `vector`, `seq` can be present at the same time

// -----

// acc.routine: same restriction per non-`none` device type — mirrors
// upstream's `acc.routine` `device_type` parallelism diagnostic.
func.func @routine_nvidia_parallelism() {
  func.return
}
"acc.routine"() <{func_name = @routine_nvidia_parallelism, sym_name = "rt_bad_nvidia", gang = [#acc.device_type<nvidia>], worker = [#acc.device_type<nvidia>]}> : () -> ()
// CHECK: only one of `gang`, `worker`, `vector`, `seq` can be present at the same time for device_type `nvidia`

// -----

// acc.routine: a `none`-device parallelism marker conflicts with any
// per-device parallelism marker — the verifier emits the per-device
// diagnostic for the conflicting non-`none` device.
func.func @routine_none_plus_nvidia() {
  func.return
}
"acc.routine"() <{func_name = @routine_none_plus_nvidia, sym_name = "rt_bad_mix", gang = [#acc.device_type<none>], worker = [#acc.device_type<nvidia>]}> : () -> ()
// CHECK: only one of `gang`, `worker`, `vector`, `seq` can be present at the same time for device_type `nvidia`

// -----

// acc.routine: a `bind(...)` entry must be a SymbolRef (`@name`) or a
// StringAttr (`"name"`). Any other attribute kind triggers BindName's
// defensive parse_attribute → not-isinstance branch.
func.func @routine_bind_bad() {
  func.return
}
acc.routine @rt_bind_bad func(@routine_bind_bad) bind(1 : i64)
// CHECK: expected SymbolRef or string attribute in bind clause

// -----

// acc.kernel_environment uses `SizedRegion<1>`: the body must be exactly
// one block. A multi-block body is rejected by the region's single-block
// constraint (independent of `NoTerminator`, which only relaxes the
// terminator requirement).
func.func @ke_multi_block() {
  "acc.kernel_environment"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> ({
  ^bb0:
    "cf.br"()[^bb1] : () -> ()
  ^bb1:
  }) : () -> ()
  func.return
}
// CHECK: Region 'region' at position 0 expected a single block, but got 2 blocks

// -----

// `SizedRegion<1>` also rejects a 0-block body. Both upstream MLIR and
// xDSL refuse the empty `({})` form.
func.func @ke_no_block() {
  "acc.kernel_environment"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> ({
  }) : () -> ()
  func.return
}
// CHECK: Region 'region' at position 0 expected a single block, but got 0 blocks

// -----

// The oilist directive lets the three clauses appear in any order, but
// each clause may appear at most once. Repeating a keyword fires the
// directive's own diagnostic.
func.func @ke_duplicate_async() {
  acc.kernel_environment async async {
    "test.op"() : () -> ()
  }
  func.return
}
// CHECK: 'async' clause specified twice

// -----

// acc.atomic.read forbids reading and writing to the same location — the
// source `x` and destination `v` operands must be distinct SSA values.
func.func @atomic_read_same(%x : memref<i32>) {
  acc.atomic.read %x = %x : memref<i32>, memref<i32>, i32
  func.return
}
// CHECK: read and write must not be to the same location for atomic reads

// -----

// acc.atomic.write checks that the pointer operand `x` dereferences to the
// value operand `expr`'s type — a nested memref<memref<...>> won't.
func.func @atomic_write_bad_type(%addr : memref<memref<i32>>, %val : i32) {
  acc.atomic.write %addr = %val : memref<memref<i32>>, i32
  func.return
}
// CHECK: address must dereference to value type

// -----

// acc.atomic.update's region argument type must match the pointee of `x`.
// A memref<i32> with an f32 block argument fails the verifier.
func.func @atomic_update_arg_type_mismatch(%x : memref<i32>, %expr : f32) {
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval: f32):
    %newval = arith.addf %xval, %expr : f32
    acc.yield %newval : f32
  }
  func.return
}
// CHECK: the type of the operand must be a pointer type whose element type is the same as that of the region argument

// -----

// acc.atomic.update's yield must produce exactly one value — the updated
// scalar.
func.func @atomic_update_multi_yield(%x : memref<i32>, %expr : i32) {
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = arith.addi %xval, %expr : i32
    acc.yield %newval, %expr : i32, i32
  }
  func.return
}
// CHECK: only updated value must be returned

// -----

// acc.atomic.update's yielded value must have the same type as the region
// argument.
func.func @atomic_update_yield_type_mismatch(%x : memref<i32>, %y : f32) {
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    acc.yield %y : f32
  }
  func.return
}
// CHECK: input and yielded value must have the same type

// -----

// acc.atomic.update's region must accept exactly one block argument.
func.func @atomic_update_too_many_args(%x : memref<i32>, %expr : i32) {
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32, %tmp: i32):
    %newval = arith.addi %xval, %expr : i32
    acc.yield %newval : i32
  }
  func.return
}
// CHECK: the region must accept exactly one argument

// -----

// acc.atomic.capture's region must contain exactly two atomic ops plus the
// implicit terminator. A single op + terminator is rejected.
func.func @atomic_capture_too_few_ops(%v : memref<i32>, %x : memref<i32>) {
  acc.atomic.capture {
    acc.atomic.read %v = %x : memref<i32>, memref<i32>, i32
  }
  func.return
}
// CHECK: expected three operations in atomic.capture region (one terminator, and two atomic ops)

// -----

// acc.atomic.capture rejects sequences other than (update,read), (read,update),
// or (read,write). Two reads in a row trips the sequence check.
func.func @atomic_capture_invalid_sequence(%v : memref<i32>, %x : memref<i32>) {
  acc.atomic.capture {
    acc.atomic.read %v = %x : memref<i32>, memref<i32>, i32
    acc.atomic.read %v = %x : memref<i32>, memref<i32>, i32
  }
  func.return
}
// CHECK: invalid sequence of operations in the capture region

// -----

// acc.atomic.capture with (update, read) requires both ops to refer to the
// same address `x`.
func.func @atomic_capture_update_read_var_mismatch(%x : memref<i32>, %y : memref<i32>, %v : memref<i32>, %expr : i32) {
  acc.atomic.capture {
    acc.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = arith.addi %xval, %expr : i32
      acc.yield %newval : i32
    }
    acc.atomic.read %v = %y : memref<i32>, memref<i32>, i32
  }
  func.return
}
// CHECK: updated variable in atomic.update must be captured in second operation

// -----

// acc.atomic.capture with (read, update) requires both ops to refer to the
// same address `x`.
func.func @atomic_capture_read_update_var_mismatch(%x : memref<i32>, %y : memref<i32>, %v : memref<i32>, %expr : i32) {
  acc.atomic.capture {
    acc.atomic.read %v = %y : memref<i32>, memref<i32>, i32
    acc.atomic.update %x : memref<i32> {
    ^bb0(%xval: i32):
      %newval = arith.addi %xval, %expr : i32
      acc.yield %newval : i32
    }
  }
  func.return
}
// CHECK: captured variable in atomic.read must be updated in second operation

// -----

// acc.atomic.capture with (read, write) requires both ops to refer to the
// same address `x`.
func.func @atomic_capture_read_write_var_mismatch(%x : memref<i32>, %y : memref<i32>, %v : memref<i32>, %expr : i32) {
  acc.atomic.capture {
    acc.atomic.read %v = %x : memref<i32>, memref<i32>, i32
    acc.atomic.write %y = %expr : memref<i32>, i32
  }
  func.return
}
// CHECK: captured variable in atomic.read must be updated in second operation
