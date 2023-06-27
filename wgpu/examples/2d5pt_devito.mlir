builtin.module attributes {gpu.container_module} {
  "gpu.module"() ({
    "gpu.func"() ({
    ^0(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : memref<260x260xf32>, %arg4 : index, %arg5 : f32, %arg6 : f32, %arg7 : f32, %arg8 : f32, %arg9 : f32, %arg10 : f32, %arg11 : memref<260x260xf32>):
      %v0 = "arith.constant"() {"value" = 2 : index} : () -> index
      %v1 = "gpu.block_id"() {"dimension" = #gpu<dim x>} : () -> index
      %v2 = "gpu.block_id"() {"dimension" = #gpu<dim y>} : () -> index
      %v3 = "gpu.thread_id"() {"dimension" = #gpu<dim x>} : () -> index
      %v4 = "gpu.thread_id"() {"dimension" = #gpu<dim y>} : () -> index
      %v5 = arith.muli %v1, %arg0 : index
      %v6 = arith.addi %v5, %arg1 : index
      %v7 = arith.muli %v2, %arg2 : index
      %v8 = arith.addi %v7, %arg1 : index
      %v9 = arith.muli %v3, %arg2 : index
      %v10 = arith.addi %v9, %arg1 : index
      %v11 = arith.muli %v4, %arg2 : index
      %v12 = arith.addi %v11, %arg1 : index
      %v13 = arith.addi %v10, %v6 : index
      %v14 = arith.addi %v12, %v8 : index
      %v15 = arith.addi %v14, %v0 : index
      %v16 = arith.addi %v13, %v0 : index
      %v17 = "memref.load"(%arg3, %v15, %v16) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %v18 = arith.addi %v14, %arg4 : index
      %v19 = arith.addi %v18, %v0 : index
      %v20 = "memref.load"(%arg3, %v19, %v16) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %v21 = arith.addi %v14, %arg2 : index
      %v22 = arith.addi %v21, %v0 : index
      %v23 = "memref.load"(%arg3, %v22, %v16) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %v24 = arith.addi %v13, %arg4 : index
      %v25 = arith.addi %v24, %v0 : index
      %v26 = "memref.load"(%arg3, %v15, %v25) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %v27 = arith.addi %v13, %arg2 : index
      %v28 = arith.addi %v27, %v0 : index
      %v29 = "memref.load"(%arg3, %v15, %v28) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %v30 = arith.mulf %v17, %arg5 : f32
      %v31 = arith.mulf %v20, %arg6 : f32
      %v32 = arith.mulf %v23, %arg6 : f32
      %v33 = arith.mulf %v17, %arg7 : f32
      %v34 = arith.addf %v31, %v32 : f32
      %v35 = arith.addf %v34, %v33 : f32
      %v36 = arith.mulf %v26, %arg6 : f32
      %v37 = arith.mulf %v29, %arg6 : f32
      %v38 = arith.addf %v36, %v37 : f32
      %v39 = arith.addf %v38, %v33 : f32
      %v40 = arith.addf %v35, %v39 : f32
      %v41 = arith.mulf %v40, %arg8 : f32
      %v42 = arith.addf %v30, %arg9 : f32
      %v43 = arith.addf %v42, %v41 : f32
      %v44 = arith.mulf %v43, %arg10 : f32
      "memref.store"(%v44, %arg11, %v15, %v16) {"nontemporal" = false} : (f32, memref<260x260xf32>, index, index) -> ()
      "gpu.return"() : () -> ()
    }) {"function_type" = (index, index, index, memref<260x260xf32>, index, f32, f32, f32, f32, f32, f32, memref<260x260xf32>) -> (), "gpu.kernel", "gpu.known_block_size" = array<i32: 128, 1, 1>, "gpu.known_grid_size" = array<i32: 2, 256, 1>, "sym_name" = "apply_kernel_kernel", "workgroup_attributions" = 0 : i64} : () -> ()
    "gpu.module_end"() : () -> ()
  }) {"sym_name" = "apply_kernel_kernel"} : () -> ()
}
