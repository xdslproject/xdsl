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
                #acc.reduction_operator<lor>
                // CHECK-SAME: #acc.reduction_operator<lor>

            ]}: () -> !acc.data_bounds_ty
                // CHECK-SAME: !acc.data_bounds_ty
