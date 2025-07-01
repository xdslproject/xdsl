// RUN: XDSL_ROUNDTRIP

builtin.module attributes {"gpu.container_module"} {
    "gpu.module"() <{"sym_name" = "gpu"}> ({
        func.func @kernel() {
            %n = arith.constant {"proc" = #gpu<processor thread_x>} 13 : index
            %one = arith.constant {"loopdim" = #gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>} 1 : index

            %memref = "memref.alloc"() {"alignment" = 0 : i64, operandSegmentSizes = array<i32: 0, 0>} : () -> memref<10x10xi32>
            %unranked = "memref.cast"(%memref) : (memref<10x10xi32>) -> memref<*xi32>
            "gpu.host_register"(%unranked) : (memref<*xi32>) -> ()
            "gpu.host_unregister"(%unranked) : (memref<*xi32>) -> ()

            %wait_token = "gpu.wait"() : () -> !gpu.async.token

            %threadidx = "gpu.thread_id"() {"dimension" = #gpu<dim x>} : () -> index
            %threadidy = "gpu.thread_id"() {"dimension" = #gpu<dim y>} : () -> index
            %threadidz = "gpu.thread_id"() {"dimension" = #gpu<dim z>} : () -> index

            %blockdimx = "gpu.block_dim"() {"dimension" = #gpu<dim x>} : () -> index
            %blockdimy = "gpu.block_dim"() {"dimension" = #gpu<dim y>} : () -> index
            %blockdimz = "gpu.block_dim"() {"dimension" = #gpu<dim z>} : () -> index

            %blockidx = "gpu.block_id"() {"dimension" = #gpu<dim x>} : () -> index
            %blockidy = "gpu.block_id"() {"dimension" = #gpu<dim y>} : () -> index
            %blockidz = "gpu.block_id"() {"dimension" = #gpu<dim z>} : () -> index

            %globalidx = "gpu.global_id"() {"dimension" = #gpu<dim x>} : () -> index
            %globalidy = "gpu.global_id"() {"dimension" = #gpu<dim y>} : () -> index
            %globalidz = "gpu.global_id"() {"dimension" = #gpu<dim z>} : () -> index

            %griddimx = "gpu.grid_dim"() {"dimension" = #gpu<dim x>} : () -> index
            %griddimy = "gpu.grid_dim"() {"dimension" = #gpu<dim y>} : () -> index
            %griddimz = "gpu.grid_dim"() {"dimension" = #gpu<dim z>} : () -> index

            %gmemref = "gpu.alloc"() {operandSegmentSizes = array<i32: 0, 0, 0>} : () -> memref<10x10xi32>
            %gdmemref = "gpu.alloc"(%griddimx, %griddimy,%griddimz) {operandSegmentSizes = array<i32: 0, 3, 0>}: (index, index, index) -> memref<?x?x?xf64>

            "gpu.memcpy"(%memref, %gmemref) {operandSegmentSizes = array<i32: 0, 1, 1>} : (memref<10x10xi32>, memref<10x10xi32>) -> ()

            "gpu.dealloc"(%gdmemref) {operandSegmentSizes = array<i32: 0, 1>} : (memref<?x?x?xf64>) -> ()

            %laneid = "gpu.lane_id"() : () -> index
            %numsubgroups = "gpu.num_subgroups"() : () -> index

            %dev = arith.constant 0 : i32
            "gpu.set_default_device"(%dev) : (i32) -> ()

            %subgroupid = "gpu.subgroup_id"() : () -> index
            %subgroupsize = "gpu.subgroup_size"() : () -> index

            %globalprodx = "gpu.all_reduce"(%globalidx) ({
            }) {"op" = #gpu<all_reduce_op mul>} : (index) -> index

            %globalsumy = "gpu.all_reduce"(%globalidy) ({
            ^bb(%lhs : index, %rhs : index):
                %sum = arith.addi %lhs, %rhs : index
                "gpu.yield"(%sum) : (index) -> ()
            }) : (index) -> index

            "gpu.launch"(%one, %one, %one, %n, %one, %one) ({
            ^bb0(%bx : index, %by : index, %bz : index,
                %tx : index, %ty : index, %tz : index,
                %num_bx : index, %num_by : index, %num_bz : index,
                %num_tx : index, %num_ty : index, %num_tz : index):
                %sum = "gpu.all_reduce"(%tx) ({
                }) {"op" = #gpu<all_reduce_op add>} : (index) -> index
                %final = arith.muli %sum, %one : index
                "gpu.terminator"() : () -> ()
            }) {operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>} : (index, index, index, index, index, index) -> ()
            "gpu.launch_func"(%n, %n, %n, %n, %n, %n, %dev, %n) {operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0>, "kernel" = @gpu::@foo} : (index, index, index, index, index, index, i32, index) -> ()

            func.return
        }
        "gpu.func"() ({
        ^bb0(%arg0: index):
            "gpu.return"() : () -> ()
        }) {"sym_name" = "foo", "kernel", "function_type" = (index) -> (), "gpu.known_block_size" = array<i32: 128, 1, 1>, "gpu.known_grid_size" = array<i32: 128, 1, 1>} : () -> ()
    }) : () -> ()
}

// CHECK:      builtin.module attributes {gpu.container_module} {
// CHECK-NEXT:     "gpu.module"() <{sym_name = "gpu"}> ({
// CHECK-NEXT:         func.func @kernel() {
// CHECK-NEXT:             %{{.*}} = arith.constant {proc = #gpu<processor thread_x>} 13 : index
// CHECK-NEXT:             %{{.*}} = arith.constant {loopdim = #gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>} 1 : index

// CHECK-NEXT:             %{{.*}} = memref.alloc() {alignment = 0 : i64} : memref<10x10xi32>
// CHECK-NEXT:             %{{.*}} = "memref.cast"(%{{.*}}) : (memref<10x10xi32>) -> memref<*xi32>
// CHECK-NEXT:             "gpu.host_register"(%{{.*}}) : (memref<*xi32>) -> ()
// CHECK-NEXT:             "gpu.host_unregister"(%{{.*}}) : (memref<*xi32>) -> ()

 // CHECK-NEXT:            %{{.*}} = "gpu.wait"() : () -> !gpu.async.token

// CHECK-NEXT:             %{{.*}} = "gpu.thread_id"() <{dimension = #gpu<dim x>}> : () -> index
// CHECK-NEXT:             %{{.*}} = "gpu.thread_id"() <{dimension = #gpu<dim y>}> : () -> index
// CHECK-NEXT:             %{{.*}} = "gpu.thread_id"() <{dimension = #gpu<dim z>}> : () -> index

// CHECK-NEXT:             %{{.*}} = "gpu.block_dim"() <{dimension = #gpu<dim x>}> : () -> index
// CHECK-NEXT:             %{{.*}} = "gpu.block_dim"() <{dimension = #gpu<dim y>}> : () -> index
// CHECK-NEXT:             %{{.*}} = "gpu.block_dim"() <{dimension = #gpu<dim z>}> : () -> index

// CHECK-NEXT:             %{{.*}} = "gpu.block_id"() <{dimension = #gpu<dim x>}> : () -> index
// CHECK-NEXT:             %{{.*}} = "gpu.block_id"() <{dimension = #gpu<dim y>}> : () -> index
// CHECK-NEXT:             %{{.*}} = "gpu.block_id"() <{dimension = #gpu<dim z>}> : () -> index

// CHECK-NEXT:             %{{.*}} = "gpu.global_id"() <{dimension = #gpu<dim x>}> : () -> index
// CHECK-NEXT:             %{{.*}} = "gpu.global_id"() <{dimension = #gpu<dim y>}> : () -> index
// CHECK-NEXT:             %{{.*}} = "gpu.global_id"() <{dimension = #gpu<dim z>}> : () -> index

// CHECK-NEXT:             %{{.*}} = "gpu.grid_dim"() <{dimension = #gpu<dim x>}> : () -> index
// CHECK-NEXT:             %{{.*}} = "gpu.grid_dim"() <{dimension = #gpu<dim y>}> : () -> index
// CHECK-NEXT:             %{{.*}} = "gpu.grid_dim"() <{dimension = #gpu<dim z>}> : () -> index

// CHECK-NEXT:             %gmemref = "gpu.alloc"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<10x10xi32>
// CHECK-NEXT:             %gdmemref = "gpu.alloc"(%griddimx, %griddimy, %griddimz) <{operandSegmentSizes = array<i32: 0, 3, 0>}> : (index, index, index) -> memref<?x?x?xf64>

// CHECK-NEXT:            "gpu.memcpy"(%memref, %gmemref) {operandSegmentSizes = array<i32: 0, 1, 1>} : (memref<10x10xi32>, memref<10x10xi32>) -> ()

// CHECK-NEXT:            "gpu.dealloc"(%gdmemref) {operandSegmentSizes = array<i32: 0, 1>} : (memref<?x?x?xf64>) -> ()

// CHECK-NEXT:             %{{.*}} = "gpu.lane_id"() : () -> index
// CHECK-NEXT:             %{{.*}} = "gpu.num_subgroups"() : () -> index

// CHECK-NEXT:             %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:             "gpu.set_default_device"(%{{.*}}) : (i32) -> ()

// CHECK-NEXT:             %{{.*}} = "gpu.subgroup_id"() : () -> index
// CHECK-NEXT:             %{{.*}} = "gpu.subgroup_size"() : () -> index

// CHECK-NEXT:             %{{.*}} = "gpu.all_reduce"(%{{.*}}) <{op = #gpu<all_reduce_op mul>}> ({
// CHECK-NEXT:             }) : (index) -> index

// CHECK-NEXT:             %{{.*}} = "gpu.all_reduce"(%{{.*}}) ({
// CHECK-NEXT:             ^{{.*}}(%{{.*}} : index, %{{.*}} : index):
// CHECK-NEXT:                 %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:                 "gpu.yield"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:             }) : (index) -> index

// CHECK-NEXT:             "gpu.launch"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}> ({
// CHECK-NEXT:             ^{{\S+}}(%{{\S+}} : index, %{{\S+}} : index, %{{\S+}} : index,
// CHECK-SAME:                 %{{\S+}} : index, %{{\S+}} : index, %{{\S+}} : index,
// CHECK-SAME:                 %{{\S+}} : index, %{{\S+}} : index, %{{\S+}} : index,
// CHECK-SAME:                 %{{\S+}} : index, %{{\S+}} : index, %{{\S+}} : index):
// CHECK-NEXT:                 %{{.*}} = "gpu.all_reduce"(%{{.*}}) <{op = #gpu<all_reduce_op add>}> ({
// CHECK-NEXT:                 }) : (index) -> index
// CHECK-NEXT:                 %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:                 "gpu.terminator"() : () -> ()
// CHECK-NEXT:             }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:             "gpu.launch_func"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{kernel = @gpu::@foo, operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0>}> : (index, index, index, index, index, index, i32, index) -> ()

// CHECK-NEXT:             func.return
// CHECK-NEXT:         }
// CHECK-NEXT:         "gpu.func"() <{function_type = (index) -> (), kernel}> ({
// CHECK-NEXT:         ^{{.*}}(%{{.*}}: index):
// CHECK-NEXT:             "gpu.return"() : () -> ()
// CHECK-NEXT:         }) {sym_name = "foo", gpu.known_block_size = array<i32: 128, 1, 1>, gpu.known_grid_size = array<i32: 128, 1, 1>} : () -> ()
// CHECK-NEXT:     }) : () -> ()

// CHECK-NEXT: }
