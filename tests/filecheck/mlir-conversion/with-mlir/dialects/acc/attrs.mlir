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
                #acc.reduction_operator<lor>
                // CHECK-SAME: #acc.reduction_operator<lor>

            ]} : () -> ()
