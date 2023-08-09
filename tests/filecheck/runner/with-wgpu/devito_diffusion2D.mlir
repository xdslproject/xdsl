// RUN: xdsl-run --wgpu %s | filecheck %s

builtin.module attributes  {"gpu.container_module"} {
  func.func @gpu_kernel(%arg0 : memref<8x8xf32>, %arg1 : memref<8x8xf32>) {
    %0 = arith.constant 9 : index
    %1 = arith.constant 4 : index
    %2 = arith.constant 1 : index
    %3 = arith.constant 0 : index
    %5, %6 = "scf.for"(%3, %0, %2, %arg0, %arg1) ({
    ^0(%arg3 : index, %arg4 : memref<8x8xf32>, %arg5 : memref<8x8xf32>):
      "gpu.launch_func"(%1, %1, %2, %2, %2, %2, %arg4, %arg5) {"kernel" = @gpu_kernel_kernel::@gpu_kernel_kernel, "operand_segment_sizes" = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 2>} : (index, index, index, index, index, index, memref<8x8xf32>, memref<8x8xf32>) -> ()
      "scf.yield"(%arg5, %arg4) : (memref<8x8xf32>, memref<8x8xf32>) -> ()
    }) : (index, index, index, memref<8x8xf32>, memref<8x8xf32>) -> (memref<8x8xf32>, memref<8x8xf32>)
    func.return
  }
  "gpu.module"() ({
    "gpu.func"() ({
    ^1(%arg0_1 : memref<8x8xf32>, %arg1_1 : memref<8x8xf32>):
      %8 = arith.constant 3 : index
      %9 = arith.constant 1 : index
      %10 = arith.constant 2 : index
      %11 = arith.constant 5.555556e-02 : f32
      %12 = arith.constant 5.000000e-01 : f32
      %13 = arith.constant -4.500000e+00 : f32
      %14 = arith.constant 2.250000e+00 : f32
      %15 = arith.constant 1.800000e+01 : f32
      %16 = "gpu.block_id"() {"dimension" = #gpu<dim x>} : () -> index
      %17 = "gpu.block_id"() {"dimension" = #gpu<dim y>} : () -> index
      %18 = arith.addi %17, %10 : index
      %19 = arith.addi %16, %10 : index
      %20 = "memref.load"(%arg0_1, %18, %19) {"nontemporal" = false} : (memref<8x8xf32>, index, index) -> f32
      %21 = arith.addi %17, %9 : index
      %22 = "memref.load"(%arg0_1, %21, %19) {"nontemporal" = false} : (memref<8x8xf32>, index, index) -> f32
      %23 = arith.addi %17, %8 : index
      %24 = "memref.load"(%arg0_1, %23, %19) {"nontemporal" = false} : (memref<8x8xf32>, index, index) -> f32
      %25 = arith.addi %16, %9 : index
      %26 = "memref.load"(%arg0_1, %18, %25) {"nontemporal" = false} : (memref<8x8xf32>, index, index) -> f32
      %27 = arith.addi %16, %8 : index
      %28 = "memref.load"(%arg0_1, %18, %27) {"nontemporal" = false} : (memref<8x8xf32>, index, index) -> f32
      %29 = arith.mulf %20, %15 : f32
      %30 = arith.mulf %22, %14 : f32
      %31 = arith.mulf %24, %14 : f32
      %32 = arith.mulf %20, %13 : f32
      %33 = arith.addf %30, %31 : f32
      %34 = arith.addf %33, %32 : f32
      %35 = arith.mulf %26, %14 : f32
      %36 = arith.mulf %28, %14 : f32
      %37 = arith.addf %35, %36 : f32
      %38 = arith.addf %37, %32 : f32
      %39 = arith.addf %34, %38 : f32
      %40 = arith.mulf %39, %12 : f32
      %41 = arith.addf %29, %40 : f32
      %42 = arith.mulf %41, %11 : f32
      "memref.store"(%42, %arg1_1, %18, %19) {"nontemporal" = false} : (f32, memref<8x8xf32>, index, index) -> ()
      "gpu.return"() : () -> ()
    }) {"function_type" = (memref<8x8xf32>, memref<8x8xf32>) -> (), "gpu.kernel", "gpu.known_block_size" = array<i32: 1, 1, 1>, "gpu.known_grid_size" = array<i32: 1, 1, 1>, "sym_name" = "gpu_kernel_kernel", "workgroup_attributions" = 0 : i64} : () -> ()
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
    "memref.store"(%init, %arg1, %four, %four) : (f32, memref<8x8xf32>, index, index) -> ()
    "func.call"(%arg1, %arg2) {"callee" = @apply_kernel} : (memref<8x8xf32>, memref<8x8xf32>) -> ()
    printf.print_format "Result : {}", %arg1 : memref<8x8xf32>
    %zero  = "arith.constant"() {"value" = 0} : () -> (index)
    "func.return"(%zero) : (index) -> ()
  }
}

// CHECK-NEXT: Result : [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.010562458075582981, 0.05379563942551613, 0.12349317967891693, 0.052650336176157, 0.0, 0.0], [0.0, 0.0, 0.05379563942551613, 0.23651812970638275, 0.4902934432029724, 0.2293916642665863, 0.0, 0.0], [0.0, 0.0, 0.12349317967891693, 0.4902934432029724, 0.9488800764083862, 0.47144585847854614, 0.0, 0.0], [0.0, 0.0, 0.052650336176157, 0.2293916642665863, 0.47144585847854614, 0.22235946357250214, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]