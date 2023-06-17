builtin.module {
  "gpu.module"() ({
    "gpu.func"() ({
    ^0(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : memref<260x260xf32>, %arg5 : index, %arg6 : f32, %arg7 : f32, %arg8 : f32, %arg9 : f32, %arg10 : f32, %arg11 : f32, %arg12 : memref<260x260xf32>):
      %0 = "gpu.block_id"() {"dimension" = #gpu<dim x>} : () -> index
      %1 = "gpu.block_id"() {"dimension" = #gpu<dim y>} : () -> index
      %2 = "gpu.block_id"() {"dimension" = #gpu<dim z>} : () -> index
      %3 = "gpu.thread_id"() {"dimension" = #gpu<dim x>} : () -> index
      %4 = "gpu.thread_id"() {"dimension" = #gpu<dim y>} : () -> index
      %5 = "gpu.thread_id"() {"dimension" = #gpu<dim z>} : () -> index
      %6 = "gpu.grid_dim"() {"dimension" = #gpu<dim x>} : () -> index
      %7 = "gpu.grid_dim"() {"dimension" = #gpu<dim y>} : () -> index
      %8 = "gpu.grid_dim"() {"dimension" = #gpu<dim z>} : () -> index
      %9 = "gpu.block_dim"() {"dimension" = #gpu<dim x>} : () -> index
      %10 = "gpu.block_dim"() {"dimension" = #gpu<dim y>} : () -> index
      %11 = "gpu.block_dim"() {"dimension" = #gpu<dim z>} : () -> index
      "cf.br"() [^1] : () -> ()
    ^1:
      %12 = arith.muli %0, %arg0 : index
      %13 = arith.addi %12, %arg1 : index
      %14 = arith.muli %1, %arg2 : index
      %15 = arith.addi %14, %arg1 : index
      %16 = arith.muli %3, %arg3 : index
      %17 = arith.addi %16, %arg1 : index
      %18 = arith.muli %4, %arg3 : index
      %19 = arith.addi %18, %arg1 : index
      %20 = arith.addi %17, %13 : index
      %21 = arith.addi %19, %15 : index
      %22 = "arith.constant"() {"value" = 2 : index} : () -> index
      %23 = arith.addi %21, %22 : index
      %24 = "arith.constant"() {"value" = 2 : index} : () -> index
      %25 = arith.addi %20, %24 : index
      %26 = "memref.load"(%arg4, %23, %25) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %27 = arith.addi %21, %arg5 : index
      %28 = "arith.constant"() {"value" = 2 : index} : () -> index
      %29 = arith.addi %27, %28 : index
      %30 = "arith.constant"() {"value" = 2 : index} : () -> index
      %31 = arith.addi %20, %30 : index
      %32 = "memref.load"(%arg4, %29, %31) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %33 = arith.addi %21, %arg3 : index
      %34 = "arith.constant"() {"value" = 2 : index} : () -> index
      %35 = arith.addi %33, %34 : index
      %36 = "arith.constant"() {"value" = 2 : index} : () -> index
      %37 = arith.addi %20, %36 : index
      %38 = "memref.load"(%arg4, %35, %37) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %39 = arith.addi %20, %arg5 : index
      %40 = "arith.constant"() {"value" = 2 : index} : () -> index
      %41 = arith.addi %21, %40 : index
      %42 = "arith.constant"() {"value" = 2 : index} : () -> index
      %43 = arith.addi %39, %42 : index
      %44 = "memref.load"(%arg4, %41, %43) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %45 = arith.addi %20, %arg3 : index
      %46 = "arith.constant"() {"value" = 2 : index} : () -> index
      %47 = arith.addi %21, %46 : index
      %48 = "arith.constant"() {"value" = 2 : index} : () -> index
      %49 = arith.addi %45, %48 : index
      %50 = "memref.load"(%arg4, %47, %49) {"nontemporal" = false} : (memref<260x260xf32>, index, index) -> f32
      %51 = arith.mulf %26, %arg6 : f32
      %52 = arith.mulf %32, %arg7 : f32
      %53 = arith.mulf %38, %arg7 : f32
      %54 = arith.mulf %26, %arg8 : f32
      %55 = arith.addf %52, %53 : f32
      %56 = arith.addf %55, %54 : f32
      %57 = arith.mulf %44, %arg7 : f32
      %58 = arith.mulf %50, %arg7 : f32
      %59 = arith.mulf %26, %arg8 : f32
      %60 = arith.addf %57, %58 : f32
      %61 = arith.addf %60, %59 : f32
      %62 = arith.addf %56, %61 : f32
      %63 = arith.mulf %62, %arg9 : f32
      %64 = arith.addf %51, %arg10 : f32
      %65 = arith.addf %64, %63 : f32
      %66 = arith.mulf %65, %arg11 : f32
      %67 = "arith.constant"() {"value" = 2 : index} : () -> index
      %68 = arith.addi %21, %67 : index
      %69 = "arith.constant"() {"value" = 2 : index} : () -> index
      %70 = arith.addi %20, %69 : index
      "memref.store"(%66, %arg12, %68, %70) {"nontemporal" = false} : (f32, memref<260x260xf32>, index, index) -> ()
      "gpu.return"() : () -> ()
    }) {"function_type" = (index, index, index, index, memref<260x260xf32>, index, f32, f32, f32, f32, f32, f32, memref<260x260xf32>) -> (), "gpu.kernel", "gpu.known_block_size" = array<i32: 32, 8, 1>, "gpu.known_grid_size" = array<i32: 8, 32, 1>, "sym_name" = "apply_kernel_kernel", "workgroup_attributions" = 0 : i64} : () -> ()
    "gpu.module_end"() : () -> ()
  }) {"sym_name" = "apply_kernel_kernel"} : () -> ()
  %71 = "arith.constant"() {"value" = 0 : i64} : () -> index
}
