// RUN: XDSL_ROUNDTRIP

"test.op"() {attrs = [
                #omp<variable_capture_kind (This)>,
                // CHECK: #omp<variable_capture_kind (This)>
                #omp<variable_capture_kind (ByRef)>,
                // CHECK-SAME: #omp<variable_capture_kind (ByRef)>
                #omp<variable_capture_kind (ByCopy)>,
                // CHECK-SAME: #omp<variable_capture_kind (ByCopy)>
                #omp<variable_capture_kind (VLAType)>,
                // CHECK-SAME: #omp<variable_capture_kind (VLAType)>

                #omp<clause_task_depend (taskdependin)>,
                // CHECK-SAME: #omp<clause_task_depend (taskdependin)
                #omp<clause_task_depend (taskdependout)>,
                // CHECK-SAME: #omp<clause_task_depend (taskdependout)
                #omp<clause_task_depend (taskdependinout)>,
                // CHECK-SAME: #omp<clause_task_depend (taskdependinout)
                #omp<clause_task_depend (taskdependmutexinoutset)>,
                // CHECK-SAME: #omp<clause_task_depend (taskdependmutexinoutset)
                #omp<clause_task_depend (taskdependinoutset)>,
                // CHECK-SAME: #omp<clause_task_depend (taskdependinoutset)

                #omp<sched_mod none>,
                // CHECK-SAME: #omp<sched_mod none>
                #omp<sched_mod monotonic>,
                // CHECK-SAME: #omp<sched_mod monotonic>
                #omp<sched_mod nonmonotonic>,
                // CHECK-SAME: #omp<sched_mod nonmonotonic>
                #omp<sched_mod simd>,
                // CHECK-SAME: #omp<sched_mod simd>

                #omp<schedulekind static>,
                // CHECK-SAME: #omp<schedulekind static>
                #omp<schedulekind dynamic>,
                // CHECK-SAME: #omp<schedulekind dynamic>
                #omp<schedulekind auto>,
                // CHECK-SAME: #omp<schedulekind auto>

                #omp<procbindkind primary>,
                // CHECK-SAME: #omp<procbindkind primary>
                #omp<procbindkind master>,
                // CHECK-SAME: #omp<procbindkind master>
                #omp<procbindkind close>,
                // CHECK-SAME: #omp<procbindkind close>
                #omp<procbindkind spread>,
                // CHECK-SAME: #omp<procbindkind spread>

                #omp<orderkind concurrent>,
                // CHECK-SAME: #omp<orderkind concurrent>

                #omp<device_type (any)>,
                // CHECK-SAME: #omp<device_type (any)>
                #omp<device_type (host)>,
                // CHECK-SAME: #omp<device_type (host)>
                #omp<device_type (nohost)>,
                // CHECK-SAME: #omp<device_type (nohost)>

                #omp<clause_requires none>,
                // CHECK-SAME: #omp<clause_requires none>
                #omp<clause_requires reverse_offload>,
                // CHECK-SAME: #omp<clause_requires reverse_offload>
                #omp<clause_requires unified_address>,
                // CHECK-SAME: #omp<clause_requires unified_address>
                #omp<clause_requires unified_shared_memory>,
                // CHECK-SAME: #omp<clause_requires unified_shared_memory>
                #omp<clause_requires dynamic_allocators>,
                // CHECK-SAME: #omp<clause_requires dynamic_allocators>

                #omp<capture_clause (to)>,
                // CHECK-SAME: #omp<capture_clause (to)>
                #omp<capture_clause (link)>,
                // CHECK-SAME: #omp<capture_clause (link)>
                #omp<capture_clause (enter)>,
                // CHECK-SAME: #omp<capture_clause (enter)>

                #omp.version<version = 11>,
                // CHECK-SAME: #omp<version <version = 11>>

                #omp.declaretarget <device_type = (any), capture_clause = (link)>,
                // CHECK-SAME: #omp<declaretarget <device_type = (any), capture_clause = (link)>>

                #omp.data_sharing_type {type = private},
                // CHECK-SAME: #omp<data_sharing_type {type = private}>
                #omp.data_sharing_type {type = firstprivate},
                // CHECK-SAME: #omp<data_sharing_type {type = firstprivate}>

                #omp<order_mod reproducible>,
                // CHECK-SAME: #omp<order_mod reproducible>
                #omp<order_mod unconstrained>,
                // CHECK-SAME: #omp<order_mod unconstrained>

                #omp<reduction_modifier (defaultmod)>,
                // CHECK-SAME: #omp<reduction_modifier (defaultmod)>
                #omp<reduction_modifier (inscan)>,
                // CHECK-SAME: #omp<reduction_modifier (inscan)>
                #omp<reduction_modifier (task)>
                // CHECK-SAME: #omp<reduction_modifier (task)>


            ]}: () -> !omp.map_bounds_ty
                // CHECK-SAME: !omp.map_bounds_ty
