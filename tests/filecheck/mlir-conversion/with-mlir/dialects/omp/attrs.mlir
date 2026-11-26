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

"test.op"() {attrs = [
    #omp<kernel_type(bare)>,
    // CHECK: #omp<kernel_type (bare)>
    #omp<kernel_type(generic)>,
    // CHECK-SAME: #omp<kernel_type (generic)>
    #omp<kernel_type(spmd)>,
    // CHECK-SAME: #omp<kernel_type (spmd)>
    #omp<kernel_type(spmd_no_loop)>,
    // CHECK-SAME: #omp<kernel_type (spmd_no_loop)>
    #omp<access_group_modifier(cgroup)>,
    // CHECK-SAME: #omp<access_group_modifier (cgroup)>
    #omp<fallback_modifier(abort)>,
    // CHECK-SAME: #omp<fallback_modifier (abort)>
    #omp<fallback_modifier(null)>,
    // CHECK-SAME: #omp<fallback_modifier (null)>
    #omp<fallback_modifier(default_mem)>,
    // CHECK-SAME: #omp<fallback_modifier (default_mem)>
    !omp.iterated<memref<i32>>,
    // CHECK-SAME: !omp.iterated<memref<i32>>
    !omp.affinity_entry_ty<memref<i32>, i64>,
    // CHECK-SAME: !omp.affinity_entry_ty<memref<i32>, i64>
    !omp.iterated<!omp.affinity_entry_ty<memref<i32>, index>>
    // CHECK-SAME: !omp.iterated<!omp.affinity_entry_ty<memref<i32>, index>>
]} : () -> ()
