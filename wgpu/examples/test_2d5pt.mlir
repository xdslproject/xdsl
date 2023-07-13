builtin.module attributes {gpu.container_module} {
  "gpu.module"() ({
    "gpu.func"() ({
    ^0(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : memref<260x260xf32>, %arg4 : index, %arg5 : f32, %arg6 : f32, %arg7 : f32, %arg8 : f32, %arg9 : f32, %arg10 : f32, %arg11 : memref<260x260xf32>):
      %0 = "arith.constant"() {"value" = 2 : index} : () -> index
      %1 = "gpu.block_id"() {"dimension" = #gpu<dim x>} : () -> index
      %2 = "gpu.block_id"() {"dimension" = #gpu<dim y>} : () -> index
      %3 = "gpu.thread_id"() {"dimension" = #gpu<dim x>} : () -> index
      %4 = "gpu.thread_id"() {"dimension" = #gpu<dim y>} : () -> index
      %5 = arith.muli %1, %arg0 : index
      %6 = arith.addi %5, %arg1 : index
      %7 = arith.muli %2, %arg2 : index
      %8 = arith.addi %7, %arg1 : index
      %9 = arith.muli %3, %arg2 : index
      %10 = arith.addi %9, %arg1 : index
      %11 = arith.muli %4, %arg2 : index
      %12 = arith.addi %11, %arg1 : index
      %13 = arith.addi %10, %6 : index
      %14 = arith.addi %12, %8 : index
      %15 = arith.addi %14, %0 : index
      %16 = arith.addi %13, %0 : index
      %17 = "memref.load"(%arg3, %15, %16) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %18 = arith.addi %14, %arg4 : index
      %19 = arith.addi %18, %0 : index
      %20 = "memref.load"(%arg3, %19, %16) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %21 = arith.addi %14, %arg2 : index
      %22 = arith.addi %21, %0 : index
      %23 = "memref.load"(%arg3, %22, %16) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %24 = arith.addi %13, %arg4 : index
      %25 = arith.addi %24, %0 : index
      %26 = "memref.load"(%arg3, %15, %25) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %27 = arith.addi %13, %arg2 : index
      %28 = arith.addi %27, %0 : index
      %29 = "memref.load"(%arg3, %15, %28) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %30 = arith.mulf %17, %arg5 : f32
      %31 = arith.mulf %20, %arg6 : f32
      %32 = arith.mulf %23, %arg6 : f32
      %33 = arith.mulf %17, %arg7 : f32
      %34 = arith.addf %31, %32 : f32
      %35 = arith.addf %34, %33 : f32
      %36 = arith.mulf %26, %arg6 : f32
      %37 = arith.mulf %29, %arg6 : f32
      %38 = arith.addf %36, %37 : f32
      %temp = arith.addf %38, %33 : f32
      %40 = arith.addf %35, %temp : f32
      %41 = arith.mulf %40, %arg8 : f32
      %42 = arith.addf %30, %arg9 : f32
      %43 = arith.addf %42, %41 : f32
      %44 = arith.mulf %43, %arg10 : f32
      "memref.store"(%44, %arg11, %15, %16) {"nontemporal" = false} : (f32, memref<260x260xf32>, index, index) -> ()
      "gpu.return"() : () -> ()
    }) {"function_type" = (index, index, index, memref<260x260xf32>, index, f32, f32, f32, f32, f32, f32, memref<260x260xf32>) -> (), "gpu.kernel", "gpu.known_block_size" = array<i32: 128, 1, 1>, "gpu.known_grid_size" = array<i32: 2, 256, 1>, "sym_name" = "apply_kernel_kernel", "workgroup_attributions" = 0 : i64} : () -> ()
    "gpu.module_end"() : () -> ()
  }) {"sym_name" = "apply_kernel_kernel"} : () -> ()
}
