builtin.module {
    "gpu.module"() ({
        func.func @main(%0 : memref<?xi32>, %1 : memref<?xi32>) {
        %i = "gpu.global_id"() {"dimension" = #gpu<dim x>} : () -> index
        %val = "memref.load"(%0, %i) : (memref<?xi32>, index) -> i32
        "memref.store"(%val, %1, %i) : (i32, memref<?xi32>, index) -> ()
        }
        "gpu.module_end"() : () -> ()
    }) {"sym_name" = "gpu_module"}: () -> ()
}
