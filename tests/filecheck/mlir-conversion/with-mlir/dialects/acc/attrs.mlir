// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

// Proves xDSL emits each `acc` dialect attribute in a spelling that
// upstream `mlir-opt` accepts and re-emits identically. The attrs are
// hung off an unregistered `"test.op"` so that no concrete op is needed
// to exercise them — `--allow-unregistered-dialect` (set in the
// MLIR_ROUNDTRIP substitution) lets mlir-opt parse the carrier op.
//
// Pre-existing attrs (`device_type`, `defaultvalue`, `data_clause`,
// `data_clause_modifier`, `variable_type_category`) currently have no
// MLIR interop test of their own — they were only covered indirectly
// through the data-clause ops. Including them here gives every `acc`
// attribute its own MLIR-interop CHECK and makes `attrs.mlir` the
// canonical place to extend coverage when new attributes land.
"test.op"() {attrs = [
                #acc.device_type<none>,
                // CHECK: #acc.device_type<none>
                #acc.device_type<nvidia>,
                // CHECK-SAME: #acc.device_type<nvidia>

                #acc<defaultvalue none>,
                // CHECK-SAME: #acc<defaultvalue none>
                #acc<defaultvalue present>,
                // CHECK-SAME: #acc<defaultvalue present>

                #acc<data_clause acc_copyin>,
                // CHECK-SAME: #acc<data_clause acc_copyin>
                #acc<data_clause acc_copyin_readonly>,
                // CHECK-SAME: #acc<data_clause acc_copyin_readonly>
                #acc<data_clause acc_copyout>,
                // CHECK-SAME: #acc<data_clause acc_copyout>
                #acc<data_clause acc_create_zero>,
                // CHECK-SAME: #acc<data_clause acc_create_zero>
                #acc<data_clause acc_cache_readonly>,
                // CHECK-SAME: #acc<data_clause acc_cache_readonly>

                #acc<data_clause_modifier none>,
                // CHECK-SAME: #acc<data_clause_modifier none>
                #acc<data_clause_modifier readonly>,
                // CHECK-SAME: #acc<data_clause_modifier readonly>
                // Multi-bit sets parse in any order but print in the
                // declaration order of the enum (zero before readonly).
                #acc<data_clause_modifier zero,readonly>,
                // CHECK-SAME: #acc<data_clause_modifier zero,readonly>
                // Note: combining `alwaysin,alwaysout` triggers upstream's
                // group alias `always` on re-emit, which xDSL doesn't model.
                // Stick to single-bit cases here; the full multi-bit
                // coverage stays in `tests/filecheck/dialects/acc/attrs.mlir`.
                #acc<data_clause_modifier alwaysin>,
                // CHECK-SAME: #acc<data_clause_modifier alwaysin>
                #acc<data_clause_modifier capture>,
                // CHECK-SAME: #acc<data_clause_modifier capture>

                // `#acc<variable_type_category ...>` is xDSL-only — upstream
                // models VariableTypeCategory as a type-categorization bit
                // enum, not a registered attribute, so mlir-opt does not
                // accept it. Coverage stays in `tests/filecheck/dialects/acc/attrs.mlir`.

                #acc.reduction_operator<none>,
                // CHECK-SAME: #acc.reduction_operator<none>
                #acc.reduction_operator<add>,
                // CHECK-SAME: #acc.reduction_operator<add>
                #acc.reduction_operator<mul>,
                // CHECK-SAME: #acc.reduction_operator<mul>
                #acc.reduction_operator<max>,
                // CHECK-SAME: #acc.reduction_operator<max>
                #acc.reduction_operator<min>,
                // CHECK-SAME: #acc.reduction_operator<min>
                #acc.reduction_operator<iand>,
                // CHECK-SAME: #acc.reduction_operator<iand>
                #acc.reduction_operator<ior>,
                // CHECK-SAME: #acc.reduction_operator<ior>
                #acc.reduction_operator<xor>,
                // CHECK-SAME: #acc.reduction_operator<xor>
                #acc.reduction_operator<eqv>,
                // CHECK-SAME: #acc.reduction_operator<eqv>
                #acc.reduction_operator<neqv>,
                // CHECK-SAME: #acc.reduction_operator<neqv>
                #acc.reduction_operator<land>,
                // CHECK-SAME: #acc.reduction_operator<land>
                #acc.reduction_operator<lor>,
                // CHECK-SAME: #acc.reduction_operator<lor>

                // Gang arg type attribute — `#acc.gang_arg_type<...>`
                // dot form (matching upstream's
                // `assemblyFormat = "`<` $value `>`"`).
                #acc.gang_arg_type<Num>,
                // CHECK-SAME: #acc.gang_arg_type<Num>
                #acc.gang_arg_type<Dim>,
                // CHECK-SAME: #acc.gang_arg_type<Dim>
                #acc.gang_arg_type<Static>,
                // CHECK-SAME: #acc.gang_arg_type<Static>

                // Variable-name metadata attribute. Upstream uses it
                // purely as a discardable attribute on non-acc ops
                // (e.g. `memref.alloca`) to preserve source-level
                // variable names through transforms. Including it on
                // the carrier `"test.op"` is enough to prove the
                // xdsl `<"..."` spelling round-trips through `mlir-opt`.
                #acc.var_name<"foo">,
                // CHECK-SAME: #acc.var_name<"foo">
                #acc.var_name<"a_longer_variable_name">,
                // CHECK-SAME: #acc.var_name<"a_longer_variable_name">

                // Parallelism-level attribute. Standalone dot form
                // (matching upstream's `EnumAttr` default
                // `assemblyFormat = "`<` $value `>`"`); the inline
                // spelling without the `#acc.par_level` prefix is
                // exercised through `#acc.specialized_routine` below.
                #acc.par_level<seq>,
                // CHECK-SAME: #acc.par_level<seq>
                #acc.par_level<gang_dim1>,
                // CHECK-SAME: #acc.par_level<gang_dim1>
                #acc.par_level<gang_dim2>,
                // CHECK-SAME: #acc.par_level<gang_dim2>
                #acc.par_level<gang_dim3>,
                // CHECK-SAME: #acc.par_level<gang_dim3>
                #acc.par_level<worker>,
                // CHECK-SAME: #acc.par_level<worker>
                #acc.par_level<vector>,
                // CHECK-SAME: #acc.par_level<vector>

                // Routine-info metadata attribute. Upstream attaches it
                // as a discardable attribute on `func.func` declarations
                // referenced by an `acc routine` directive. Carrying it
                // here proves the xdsl `<[...]>` spelling is bit-
                // compatible with upstream `mlir-opt`. Note: upstream's
                // tablegen-generated parser rejects `<[]>` — the array
                // must be non-empty — so the empty case lives in
                // `tests/dialects/test_acc.py` (Python API) instead.
                #acc.routine_info<[@rt1]>,
                // CHECK-SAME: #acc.routine_info<[@rt1]>
                #acc.routine_info<[@rt_gang, @rt_vector]>,
                // CHECK-SAME: #acc.routine_info<[@rt_gang, @rt_vector]>

                // Specialized-routine metadata attribute. Upstream
                // attaches this to device-specialized `func.func`s
                // produced by the routine-specialization pass. The
                // `$level` slot is the `par_level` enum and prints
                // inline as `<value>` (no `#acc.par_level` prefix).
                #acc.specialized_routine<@rt_gang, <gang_dim1>, "foo">,
                // CHECK-SAME: #acc.specialized_routine<@rt_gang, <gang_dim1>, "foo">
                #acc.specialized_routine<@rt_vector, <vector>, "foo">,
                // CHECK-SAME: #acc.specialized_routine<@rt_vector, <vector>, "foo">
                #acc.specialized_routine<@scope::@rt_worker, <worker>, "bar">,
                // CHECK-SAME: #acc.specialized_routine<@scope::@rt_worker, <worker>, "bar">

                // Declare-clause metadata attribute. Upstream attaches
                // it to a variable's creation site (e.g. `memref.global`,
                // `llvm.mlir.global`) to advertise which user-level
                // `acc declare` clause produced the capture. Upstream's
                // `struct(params)` printer adds extra whitespace around
                // the `DataClauseAttr` value (`dataClause =  acc_create`)
                // — xDSL emits the cleaner single-space form
                // (`dataClause = acc_create`), and the final `xdsl-opt`
                // re-emission in the round-trip chain produces the
                // single-space form regardless of which spelling
                // `mlir-opt` reflected back.
                #acc.declare<dataClause = acc_create>,
                // CHECK-SAME: #acc.declare<dataClause = acc_create>
                #acc.declare<dataClause = acc_copyin, implicit = true>,
                // CHECK-SAME: #acc.declare<dataClause = acc_copyin, implicit = true>
                // `implicit = false` round-trips to the elided form on
                // both sides — `DefaultValuedParameter<"bool", "false">`
                // upstream and the matching Python-side default.
                #acc.declare<dataClause = acc_copyout, implicit = false>,
                // CHECK-SAME: #acc.declare<dataClause = acc_copyout>
                // Field-reorder on parse: both dialects accept any
                // order and re-emit in the upstream declaration order
                // (`dataClause`, `implicit`).
                #acc.declare<implicit = true, dataClause = acc_declare_device_resident>,
                // CHECK-SAME: #acc.declare<dataClause = acc_declare_device_resident, implicit = true>

                // Declare-action metadata attribute. Upstream attaches
                // it to the variable's allocation site so the declare-
                // directive lowering can find the pre/post (de)alloc
                // hooks. All four slots are independently optional
                // (`OptionalParameter<SymbolRefAttr>`), so the empty
                // `<>` form is bit-compatible across both dialects.
                #acc.declare_action<>,
                // CHECK-SAME: #acc.declare_action<>
                #acc.declare_action<postAlloc = @post_alloc_hook>,
                // CHECK-SAME: #acc.declare_action<postAlloc = @post_alloc_hook>
                // Nested symref slots round-trip identically.
                #acc.declare_action<preAlloc = @scope::@pre_alloc_hook>,
                // CHECK-SAME: #acc.declare_action<preAlloc = @scope::@pre_alloc_hook>
                #acc.declare_action<preAlloc = @a, postAlloc = @b, preDealloc = @c, postDealloc = @d>,
                // CHECK-SAME: #acc.declare_action<preAlloc = @a, postAlloc = @b, preDealloc = @c, postDealloc = @d>
                // Subset + reordered input: both dialects re-emit in
                // declaration order (preAlloc, postAlloc, preDealloc,
                // postDealloc), dropping absent slots.
                #acc.declare_action<postDealloc = @d, preAlloc = @a>,
                // CHECK-SAME: #acc.declare_action<preAlloc = @a, postDealloc = @d>

                // Combined constructs attribute. Upstream's `EnumAttr`
                // default produces the dot form `#acc.combined_constructs<...>`
                // — *not* the spaced opaque form
                // `#acc<combined_constructs ...>` — so xDSL must emit the
                // dot form for `mlir-opt` to round-trip identically.
                #acc.combined_constructs<kernels_loop>,
                // CHECK-SAME: #acc.combined_constructs<kernels_loop>
                #acc.combined_constructs<parallel_loop>,
                // CHECK-SAME: #acc.combined_constructs<parallel_loop>
                #acc.combined_constructs<serial_loop>
                // CHECK-SAME: #acc.combined_constructs<serial_loop>

            ]} : () -> ()
