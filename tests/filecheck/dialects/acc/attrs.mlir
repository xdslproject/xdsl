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
                #acc<variable_type_category array,composite>
                // CHECK-SAME: #acc<variable_type_category array,composite>

            ]}: () -> !acc.data_bounds_ty
                // CHECK-SAME: !acc.data_bounds_ty
