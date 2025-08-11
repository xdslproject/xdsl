// RUN: xdsl-opt -t wgsl %s | filecheck %s

builtin.module attributes {gpu.container_module} {
  "gpu.module"() ({
    "gpu.func"() ({
    ^bb0(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : memref<260x260xf32>, %arg4 : index, %arg5 : f32, %arg6 : f32, %arg7 : f32, %arg8 : f32, %arg9 : f32, %arg10 : f32, %arg11 : memref<260x260xf32>, %arg12: memref<260x260xindex>):
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
    }) {"function_type" = (index, index, index, memref<260x260xf32>, index, f32, f32, f32, f32, f32, f32, memref<260x260xf32>, memref<260x260xindex>) -> (),
        "gpu.kernel", "gpu.known_block_size" = array<i32: 128, 1, 1>, "gpu.known_grid_size" = array<i32: 2, 256, 1>,
        "sym_name" = "apply_kernel_kernel",
        "workgroup_attributions" = 0 : i64
       } : () -> ()
  }) {"sym_name" = "apply_kernel_kernel"} : () -> ()
}

// CHECK:           @group(0) @binding(0)
// CHECK-NEXT:      var<storage,read> varg0: u32;

// CHECK:           @group(0) @binding(1)
// CHECK-NEXT:      var<storage,read> varg1: u32;

// CHECK:           @group(0) @binding(2)
// CHECK-NEXT:      var<storage,read> varg2: u32;

// CHECK:           @group(0) @binding(3)
// CHECK-NEXT:      var<storage,read> varg3: array<f32>;

// CHECK:           @group(0) @binding(4)
// CHECK-NEXT:      var<storage,read> varg4: u32;

// CHECK:           @group(0) @binding(5)
// CHECK-NEXT:      var<storage,read> varg5: f32;

// CHECK:           @group(0) @binding(6)
// CHECK-NEXT:      var<storage,read> varg6: f32;

// CHECK:           @group(0) @binding(7)
// CHECK-NEXT:      var<storage,read> varg7: f32;

// CHECK:           @group(0) @binding(8)
// CHECK-NEXT:      var<storage,read> varg8: f32;

// CHECK:           @group(0) @binding(9)
// CHECK-NEXT:      var<storage,read> varg9: f32;

// CHECK:           @group(0) @binding(10)
// CHECK-NEXT:      var<storage,read> varg10: f32;

// CHECK:           @group(0) @binding(11)
// CHECK-NEXT:      var<storage,read_write> varg11: array<f32>;

// CHECK:           @group(0) @binding(12)
// CHECK-NEXT:      var<storage,read> varg12: array<u32>;

// CHECK:           @compute
// CHECK-NEXT:      @workgroup_size(128,1,1)
// CHECK-NEXT:      fn apply_kernel_kernel(@builtin(global_invocation_id) global_invocation_id : vec3<u32>,
// CHECK-NEXT:      @builtin(workgroup_id) workgroup_id : vec3<u32>,
// CHECK-NEXT:      @builtin(local_invocation_id) local_invocation_id : vec3<u32>,
// CHECK-NEXT:      @builtin(num_workgroups) num_workgroups : vec3<u32>) {

// CHECK:               let v0 : u32 = 2u;
// CHECK-NEXT:          let v1: u32 = workgroup_id.x;
// CHECK-NEXT:          let v2: u32 = workgroup_id.y;
// CHECK-NEXT:          let v3: u32 = local_invocation_id.x;
// CHECK-NEXT:          let v4: u32 = local_invocation_id.y;
// CHECK-NEXT:          let v5 = v1 * varg0;
// CHECK-NEXT:          let v6 = v5 + varg1;
// CHECK-NEXT:          let v7 = v2 * varg2;
// CHECK-NEXT:          let v8 = v7 + varg1;
// CHECK-NEXT:          let v9 = v3 * varg2;
// CHECK-NEXT:          let v10 = v9 + varg1;
// CHECK-NEXT:          let v11 = v4 * varg2;
// CHECK-NEXT:          let v12 = v11 + varg1;
// CHECK-NEXT:          let v13 = v10 + v6;
// CHECK-NEXT:          let v14 = v12 + v8;
// CHECK-NEXT:          let v15 = v14 + v0;
// CHECK-NEXT:          let v16 = v13 + v0;
// CHECK-NEXT:          let v17 = varg3[260u * v15 + 1u * v16];
// CHECK-NEXT:          let v18 = v14 + varg4;
// CHECK-NEXT:          let v19 = v18 + v0;
// CHECK-NEXT:          let v20 = varg3[260u * v19 + 1u * v16];
// CHECK-NEXT:          let v21 = v14 + varg2;
// CHECK-NEXT:          let v22 = v21 + v0;
// CHECK-NEXT:          let v23 = varg3[260u * v22 + 1u * v16];
// CHECK-NEXT:          let v24 = v13 + varg4;
// CHECK-NEXT:          let v25 = v24 + v0;
// CHECK-NEXT:          let v26 = varg3[260u * v15 + 1u * v25];
// CHECK-NEXT:          let v27 = v13 + varg2;
// CHECK-NEXT:          let v28 = v27 + v0;
// CHECK-NEXT:          let v29 = varg3[260u * v15 + 1u * v28];
// CHECK-NEXT:          let v30 = v17 * varg5;
// CHECK-NEXT:          let v31 = v20 * varg6;
// CHECK-NEXT:          let v32 = v23 * varg6;
// CHECK-NEXT:          let v33 = v17 * varg7;
// CHECK-NEXT:          let v34 = v31 + v32;
// CHECK-NEXT:          let v35 = v34 + v33;
// CHECK-NEXT:          let v36 = v26 * varg6;
// CHECK-NEXT:          let v37 = v29 * varg6;
// CHECK-NEXT:          let v38 = v36 + v37;
// CHECK-NEXT:          let vtemp = v38 + v33;
// CHECK-NEXT:          let v39 = v35 + vtemp;
// CHECK-NEXT:          let v40 = v39 * varg8;
// CHECK-NEXT:          let v41 = v30 + varg9;
// CHECK-NEXT:          let v42 = v41 + v40;
// CHECK-NEXT:          let v43 = v42 * varg10;
// CHECK-NEXT:          varg11[260u * v15 + 1u * v16] = v43;
// CHECK-NEXT:              }
