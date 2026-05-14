// RUN: XDSL_ROUNDTRIP

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
                // declaration order of the StrEnum (zero before readonly).
                #acc<data_clause_modifier readonly,zero>,
                // CHECK-SAME: #acc<data_clause_modifier zero,readonly>
                #acc<data_clause_modifier alwaysin,alwaysout,capture>,
                // CHECK-SAME: #acc<data_clause_modifier alwaysin,alwaysout,capture>

                #acc<variable_type_category uncategorized>,
                // CHECK-SAME: #acc<variable_type_category uncategorized>
                #acc<variable_type_category scalar>,
                // CHECK-SAME: #acc<variable_type_category scalar>
                #acc<variable_type_category array,composite>,
                // CHECK-SAME: #acc<variable_type_category array,composite>

                // Reduction operator attribute. Prints with `<value>`
                // surrounding the parameter (matching upstream MLIR's
                // `assemblyFormat = "`<` $value `>`"`), so when consumed
                // inline via `$reductionOperator` in a future op format the
                // spelling is `reduction_operator <add>` rather than the
                // long-form `#acc.reduction_operator<add>`.
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

                // Gang arg type attribute. Same `<value>` printer style as
                // `reduction_operator` — when later consumed inline via
                // `$gangOperandsArgType` in `acc.loop`'s assembly format
                // the spelling is just `<Num>` rather than the long-form
                // `#acc.gang_arg_type<Num>`.
                #acc.gang_arg_type<Num>,
                // CHECK-SAME: #acc.gang_arg_type<Num>
                #acc.gang_arg_type<Dim>,
                // CHECK-SAME: #acc.gang_arg_type<Dim>
                #acc.gang_arg_type<Static>,
                // CHECK-SAME: #acc.gang_arg_type<Static>

                // Variable-name metadata attribute. Upstream uses it
                // purely as a discardable attribute on non-acc ops
                // (e.g. `memref.alloca`) to preserve source-level
                // variable names through transforms. Included here so
                // the xdsl-only round-trip is exercised alongside
                // every other `acc.*` attribute.
                #acc.var_name<"foo">,
                // CHECK-SAME: #acc.var_name<"foo">
                #acc.var_name<"a_longer_variable_name">,
                // CHECK-SAME: #acc.var_name<"a_longer_variable_name">

                // Parallelism-level attribute. Consumed inline as the
                // `$level` parameter of `#acc.specialized_routine` and
                // also reachable standalone (dot form
                // `#acc.par_level<value>`).
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
                // referenced by an `acc routine` directive — the array
                // points back at the matching `acc.routine` symbols.
                #acc.routine_info<[]>,
                // CHECK-SAME: #acc.routine_info<[]>
                #acc.routine_info<[@rt1]>,
                // CHECK-SAME: #acc.routine_info<[@rt1]>
                #acc.routine_info<[@rt_gang, @rt_vector]>,
                // CHECK-SAME: #acc.routine_info<[@rt_gang, @rt_vector]>

                // Specialized-routine metadata attribute. Attached to
                // the device-specialized `func.func` produced by the
                // routine-specialization pass. The `$level` slot prints
                // inline as `<value>` (no `#acc.par_level` prefix), so
                // the on-the-wire form is
                // `<@routine, <level>, "origname">`.
                #acc.specialized_routine<@rt_gang, <gang_dim1>, "foo">,
                // CHECK-SAME: #acc.specialized_routine<@rt_gang, <gang_dim1>, "foo">
                #acc.specialized_routine<@rt_vector, <vector>, "foo">,
                // CHECK-SAME: #acc.specialized_routine<@rt_vector, <vector>, "foo">
                #acc.specialized_routine<@scope::@rt_worker, <worker>, "bar">,
                // CHECK-SAME: #acc.specialized_routine<@scope::@rt_worker, <worker>, "bar">

                // Declare-clause metadata attribute. Upstream attaches it
                // to a variable's creation site (e.g. `memref.global`,
                // `llvm.mlir.global`, allocations) to advertise which
                // user-level `acc declare` clause produced the capture.
                // `implicit` defaults to `false` and is elided in that
                // case; struct-style fields parse in any order but print
                // in the upstream declaration order (`dataClause`,
                // `implicit`).
                #acc.declare<dataClause = acc_create>,
                // CHECK-SAME: #acc.declare<dataClause = acc_create>
                #acc.declare<dataClause = acc_copyin, implicit = true>,
                // CHECK-SAME: #acc.declare<dataClause = acc_copyin, implicit = true>
                // `implicit = false` round-trips to the elided form.
                #acc.declare<dataClause = acc_copyout, implicit = false>,
                // CHECK-SAME: #acc.declare<dataClause = acc_copyout>
                // Field order is not load-bearing on parse.
                #acc.declare<implicit = true, dataClause = acc_declare_device_resident>,
                // CHECK-SAME: #acc.declare<dataClause = acc_declare_device_resident, implicit = true>

                // Declare-action metadata attribute. Upstream attaches
                // it to the variable's allocation site so the declare-
                // directive lowering can find the pre/post (de)alloc
                // hooks for the variable. Each of the four slots is
                // independently optional; absent slots are stored as
                // `NoneAttr` and elided on print. The all-empty form
                // is `<>` (matches upstream).
                #acc.declare_action<>,
                // CHECK-SAME: #acc.declare_action<>
                #acc.declare_action<preAlloc = @pre_alloc_hook>,
                // CHECK-SAME: #acc.declare_action<preAlloc = @pre_alloc_hook>
                #acc.declare_action<postAlloc = @scope::@post_alloc_hook>,
                // CHECK-SAME: #acc.declare_action<postAlloc = @scope::@post_alloc_hook>
                #acc.declare_action<preAlloc = @a, postAlloc = @b, preDealloc = @c, postDealloc = @d>,
                // CHECK-SAME: #acc.declare_action<preAlloc = @a, postAlloc = @b, preDealloc = @c, postDealloc = @d>
                // Subset + reordered input prints back in declaration
                // order (preAlloc, postAlloc, preDealloc, postDealloc).
                #acc.declare_action<postDealloc = @d, preAlloc = @a>,
                // CHECK-SAME: #acc.declare_action<preAlloc = @a, postDealloc = @d>

                // Combined constructs attribute. Used by `acc.loop` to
                // identify the user-level `kernels loop` / `parallel loop`
                // / `serial loop` it was decomposed from.
                #acc.combined_constructs<kernels_loop>,
                // CHECK-SAME: #acc.combined_constructs<kernels_loop>
                #acc.combined_constructs<parallel_loop>,
                // CHECK-SAME: #acc.combined_constructs<parallel_loop>
                #acc.combined_constructs<serial_loop>
                // CHECK-SAME: #acc.combined_constructs<serial_loop>

            ]}: () -> !acc.data_bounds_ty
                // CHECK-SAME: !acc.data_bounds_ty
