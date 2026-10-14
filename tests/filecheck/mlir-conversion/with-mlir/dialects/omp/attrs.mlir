// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

// Proves xDSL emits omp attributes in a spelling that upstream `mlir-opt`
// accepts and re-emits identically. The attributes are hung off an
// unregistered `"test.op"` so no concrete op is needed.
"test.op"() {attrs = [
                #omp<declaretarget<device_type = (any), capture_clause = (to), automap = false>>,
                // CHECK: #omp<declaretarget <device_type = (any), capture_clause = (to), automap = false>>
                #omp<declaretarget<device_type = (host), capture_clause = (enter), automap = true>>,
                // CHECK-SAME: #omp<declaretarget <device_type = (host), capture_clause = (enter), automap = true>>
                #omp<declaretarget<automap = true>>,
                // CHECK-SAME: #omp<declaretarget <automap = true>>
                #omp<declaretarget<device_type = (nohost), automap = false>>,
                // CHECK-SAME: #omp<declaretarget <device_type = (nohost), automap = false>>
                #omp<declaretarget<capture_clause = (link)>>,
                // CHECK-SAME: #omp<declaretarget <capture_clause = (link)>>
                #omp<declaretarget<>>
                // CHECK-SAME: #omp<declaretarget <>>
            ]} : () -> ()
