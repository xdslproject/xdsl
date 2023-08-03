// RUN: xdsl-run --wgpu %s | filecheck %s

builtin.module attributes  {"gpu.container_module"} {
  func.func @gpu_kernel(%arg0 : memref<8x8xf32>, %arg1 : memref<8x8xf32>) {
    %0 = arith.constant 3 : index
    %1 = arith.constant 2 : index
    %2 = arith.constant 9 : index
    %3 = arith.constant 1.800000e+01 : f32
    %4 = arith.constant 2.250000e+00 : f32
    %5 = arith.constant -4.500000e+00 : f32
    %6 = arith.constant 5.000000e-01 : f32
    %7 = arith.constant 5.555556e-02 : f32
    %8 = arith.constant 4 : index
    %9 = arith.constant 1 : index
    %10 = arith.constant 0 : index
    %11, %12 = "scf.for"(%10, %2, %9, %arg0, %arg1) ({
    ^0(%arg2 : index, %arg3 : memref<8x8xf32>, %arg4 : memref<8x8xf32>):
      printf.print_format "Launching GPU kernel"
      "gpu.launch_func"(%8, %8, %9, %9, %9, %9, %1, %arg3, %9, %0, %3, %4, %5, %6, %7, %arg4) {"kernel" = @gpu_kernel_kernel::@gpu_kernel_kernel, "operand_segment_sizes" = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 10>} : (index, index, index, index, index, index, index, memref<8x8xf32>, index, index, f32, f32, f32, f32, f32, memref<8x8xf32>) -> ()
      "scf.yield"(%arg4, %arg3) : (memref<8x8xf32>, memref<8x8xf32>) -> ()
    }) : (index, index, index, memref<8x8xf32>, memref<8x8xf32>) -> (memref<8x8xf32>, memref<8x8xf32>)
    func.return
  }
  "gpu.module"() ({
    "gpu.func"() ({
    ^1(%arg0_1 : index, %arg1_1 : memref<8x8xf32>, %arg2_1 : index, %arg3_1 : index, %arg4_1 : f32, %arg5 : f32, %arg6 : f32, %arg7 : f32, %arg8 : f32, %arg9 : memref<8x8xf32>):
      %13 = "gpu.global_id"() {"dimension" = #gpu<dim x>} : () -> index
      %14 = "gpu.global_id"() {"dimension" = #gpu<dim y>} : () -> index
      %15 = arith.addi %14, %arg0_1 : index
      %16 = arith.addi %13, %arg0_1 : index
      %17 = "memref.load"(%arg1_1, %15, %16) : (memref<8x8xf32>, index, index) -> f32
      %18 = arith.addi %14, %arg2_1 : index
      %19 = "memref.load"(%arg1_1, %18, %16) : (memref<8x8xf32>, index, index) -> f32
      %20 = arith.addi %14, %arg3_1 : index
      %21 = "memref.load"(%arg1_1, %20, %16) : (memref<8x8xf32>, index, index) -> f32
      %22 = arith.addi %13, %arg2_1 : index
      %23 = "memref.load"(%arg1_1, %15, %22) : (memref<8x8xf32>, index, index) -> f32
      %24 = arith.addi %13, %arg3_1 : index
      %25 = "memref.load"(%arg1_1, %15, %24) : (memref<8x8xf32>, index, index) -> f32
      %26 = arith.mulf %17, %arg4_1 : f32
      %27 = arith.mulf %19, %arg5 : f32
      %28 = arith.mulf %21, %arg5 : f32
      %29 = arith.mulf %17, %arg6 : f32
      %30 = arith.addf %27, %28 : f32
      %31 = arith.addf %30, %29 : f32
      %32 = arith.mulf %23, %arg5 : f32
      %33 = arith.mulf %25, %arg5 : f32
      %34 = arith.addf %32, %33 : f32
      %35 = arith.addf %34, %29 : f32
      %36 = arith.addf %31, %35 : f32
      %37 = arith.mulf %36, %arg7 : f32
      %38 = arith.addf %26, %37 : f32
      %39 = arith.mulf %38, %arg8 : f32
      %cst = "arith.constant"() {"value" = 42.0} : () -> f32
      "memref.store"(%39, %arg9, %15, %16) : (f32, memref<8x8xf32>, index, index) -> ()
      "gpu.return"() : () -> ()
    }) {"function_type" = (index, memref<8x8xf32>, index, index, f32, f32, f32, f32, f32, memref<8x8xf32>) -> (), "gpu.kernel", "gpu.known_block_size" = array<i32: 1, 1, 1>, "gpu.known_grid_size" = array<i32: 4, 4, 1>, "sym_name" = "gpu_kernel_kernel", "workgroup_attributions" = 0 : i64} : () -> ()
    "gpu.module_end"() : () -> ()
  }) {"sym_name" = "gpu_kernel_kernel"} : () -> ()
  func.func @apply_kernel(%arg0_2 : memref<8x8xf32>, %arg1_2 : memref<8x8xf32>) {
    %40 = "gpu.alloc"() {"operand_segment_sizes" = array<i32: 0, 0, 0>} : () -> memref<8x8xf32>
    "gpu.memcpy"(%40, %arg1_2) {"operand_segment_sizes" = array<i32: 0, 1, 1>} : (memref<8x8xf32>, memref<8x8xf32>) -> ()
    %41 = "gpu.alloc"() {"operand_segment_sizes" = array<i32: 0, 0, 0>} : () -> memref<8x8xf32>
    "gpu.memcpy"(%41, %arg0_2) {"operand_segment_sizes" = array<i32: 0, 1, 1>} : (memref<8x8xf32>, memref<8x8xf32>) -> ()
    "func.call"(%41, %40) {"callee" = @gpu_kernel} : (memref<8x8xf32>, memref<8x8xf32>) -> ()
    "gpu.memcpy"(%arg0_2, %41) {"operand_segment_sizes" = array<i32: 0, 1, 1>} : (memref<8x8xf32>, memref<8x8xf32>) -> ()
    "gpu.dealloc"(%41) {"operand_segment_sizes" = array<i32: 0, 1>} : (memref<8x8xf32>) -> ()
    "gpu.memcpy"(%arg1_2, %40) {"operand_segment_sizes" = array<i32: 0, 1, 1>} : (memref<8x8xf32>, memref<8x8xf32>) -> ()
    "gpu.dealloc"(%40) {"operand_segment_sizes" = array<i32: 0, 1>} : (memref<8x8xf32>) -> ()
    func.return
  }
  func.func @main() -> index {
    %init = "arith.constant"() {"value" = 5.0} : () -> f32
    %four = "arith.constant"() {"value" = 4} : () -> index
    %arg1 = "memref.alloc"() {"alignment" = 0 : i64, "operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<8x8xf32>
    %arg2 = "memref.alloc"() {"alignment" = 0 : i64, "operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<8x8xf32>
    "memref.store"(%init, %arg2, %four, %four) : (f32, memref<8x8xf32>, index, index) -> ()
    "func.call"(%arg1, %arg2) {"callee" = @apply_kernel} : (memref<8x8xf32>, memref<8x8xf32>) -> ()
    printf.print_format "Result : {}", %arg1 : memref<8x8xf32>
    %zero  = "arith.constant"() {"value" = 0} : () -> (index)
    "func.return"(%zero) : (index) -> ()
  }
}