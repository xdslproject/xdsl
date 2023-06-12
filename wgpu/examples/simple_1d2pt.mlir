builtin.module {
    "gpu.module"() ({
        func.func @main(%0 : memref<?xi32>, %1 : memref<?xi32>) {
        %i = "gpu.global_id"() {"dimension" = #gpu<dim x>} : () -> index
        %cst1 = "arith.constant"() {"value" = 1 : index} : () -> index
        %cstm1 = "arith.constant"() {"value" = -1 : index} : () -> index
        %cst2 = "arith.constant"() {"value" = 2 : i32} : () -> i32
        %im1 = "arith.addi"(%i, %cstm1) : (index, index) -> index
        %ip1 = "arith.addi"(%i, %cst1) : (index, index) -> index
        %val = "memref.load"(%0, %i) : (memref<?xi32>, index) -> i32
        %valm1 = "memref.load"(%0, %im1) : (memref<?xi32>, index) -> i32
        %valp1 = "memref.load"(%0, %ip1) : (memref<?xi32>, index) -> i32
        %sides = "arith.addi"(%valm1, %valp1) : (i32, i32) -> i32
        %val2 = "arith.muli"(%val, %cst2) : (i32, i32) -> i32
        %res = "arith.subi"(%sides, %val2) : (i32, i32) -> i32
        "memref.store"(%res, %1, %i) : (i32, memref<?xi32>, index) -> ()
        }
        "gpu.module_end"() : () -> ()
    }) {"sym_name" = "gpu_module"}: () -> ()
}