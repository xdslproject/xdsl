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
                
                #omp<orderkind concurrent>
                // CHECK-SAME: #omp<orderkind concurrent>

            ]}: () -> !omp.map_bounds_ty
                // CHECK-SAME: !omp.map_bounds_ty
