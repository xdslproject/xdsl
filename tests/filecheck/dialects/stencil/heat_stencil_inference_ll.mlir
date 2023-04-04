// RUN: xdsl-opt %s -t mlir -p stencil-shape-inference,convert-stencil-to-ll-mlir | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%data : memref<2xmemref<?x?x?xf32>>):
    %time_m = "arith.constant"() {"value" = 0 : index} : () -> index
    %time_M = "arith.constant"() {"value" = 1000 : index} : () -> index
    %step = "arith.constant"() {"value" = 1 : index} : () -> index
    %0 = "arith.constant"() {"value" = 1 : index} : () -> index
    %time_M_1 = "arith.addi"(%time_M, %0) : (index, index) -> index
    "scf.for"(%time_m, %time_M_1, %step) ({
    ^1(%time : index):
      %time_1 = "arith.index_cast"(%time) : (index) -> i64
      %1 = "arith.constant"() {"value" = 2 : i64} : () -> i64
      %2 = "arith.constant"() {"value" = 0 : i64} : () -> i64
      %3 = "arith.addi"(%time_1, %2) : (i64, i64) -> i64
      %t0 = "arith.remsi"(%3, %1) : (i64, i64) -> i64
      %4 = "arith.constant"() {"value" = 1 : i64} : () -> i64
      %5 = "arith.addi"(%time_1, %4) : (i64, i64) -> i64
      %t1 = "arith.remsi"(%5, %1) : (i64, i64) -> i64
      %t0_w_size = "arith.index_cast"(%t0) : (i64) -> index
      %t0_w_size_1 = "memref.load"(%data, %t0_w_size) : (memref<2xmemref<?x?x?xf32>>, index) -> memref<?x?x?xf32>
      %t0_w_size_2 = "stencil.external_load"(%t0_w_size_1) : (memref<?x?x?xf32>) -> !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f32>
      %t0_w_size_3 = "stencil.cast"(%t0_w_size_2) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[104 : i64, 104 : i64, 104 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f32>) -> !stencil.field<[100 : i64, 100 : i64, 100 : i64], f32>
      %t1_w_size = "arith.index_cast"(%t1) : (i64) -> index
      %t1_w_size_1 = "memref.load"(%data, %t1_w_size) : (memref<2xmemref<?x?x?xf32>>, index) -> memref<?x?x?xf32>
      %t1_w_size_2 = "stencil.external_load"(%t1_w_size_1) : (memref<?x?x?xf32>) -> !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f32>
      %t1_w_size_3 = "stencil.cast"(%t1_w_size_2) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[104 : i64, 104 : i64, 104 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f32>) -> !stencil.field<[100 : i64, 100 : i64, 100 : i64], f32>
      %6 = "stencil.load"(%t0_w_size_3) : (!stencil.field<[100 : i64, 100 : i64, 100 : i64], f32>) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f32>
      %7 = "stencil.apply"(%6) ({
      ^2(%t0_buff : !stencil.temp<[-1 : i64], f32>):
        %8 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %9 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[-1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %10 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %11 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[-2 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %12 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[2 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %13 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, -1 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %14 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, 1 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %15 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, -2 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %16 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, 2 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %17 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, 0 : i64, -1 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %18 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, 0 : i64, 1 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %19 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, 0 : i64, -2 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %20 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, 0 : i64, 2 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %dt = "arith.constant"() {"value" = 4.122440608513459e-06 : f32} : () -> f32
        %21 = "arith.constant"() {"value" = -1 : i64} : () -> i64
        %22 = "math.fpowi"(%dt, %21) : (f32, i64) -> f32
        %23 = "arith.mulf"(%22, %8) : (f32, f32) -> f32
        %24 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
        %h_x = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %25 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %26 = "math.fpowi"(%h_x, %25) : (f32, i64) -> f32
        %27 = "arith.mulf"(%24, %26) : (f32, f32) -> f32
        %28 = "arith.mulf"(%27, %9) : (f32, f32) -> f32
        %29 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
        %h_x_1 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %30 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %31 = "math.fpowi"(%h_x_1, %30) : (f32, i64) -> f32
        %32 = "arith.mulf"(%29, %31) : (f32, f32) -> f32
        %33 = "arith.mulf"(%32, %10) : (f32, f32) -> f32
        %34 = "arith.constant"() {"value" = -2.5 : f32} : () -> f32
        %h_x_2 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %35 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %36 = "math.fpowi"(%h_x_2, %35) : (f32, i64) -> f32
        %37 = "arith.mulf"(%34, %36) : (f32, f32) -> f32
        %38 = "arith.mulf"(%37, %8) : (f32, f32) -> f32
        %39 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
        %h_x_3 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %40 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %41 = "math.fpowi"(%h_x_3, %40) : (f32, i64) -> f32
        %42 = "arith.mulf"(%39, %41) : (f32, f32) -> f32
        %43 = "arith.mulf"(%42, %11) : (f32, f32) -> f32
        %44 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
        %h_x_4 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %45 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %46 = "math.fpowi"(%h_x_4, %45) : (f32, i64) -> f32
        %47 = "arith.mulf"(%44, %46) : (f32, f32) -> f32
        %48 = "arith.mulf"(%47, %12) : (f32, f32) -> f32
        %49 = "arith.addf"(%28, %33) : (f32, f32) -> f32
        %50 = "arith.addf"(%49, %38) : (f32, f32) -> f32
        %51 = "arith.addf"(%50, %43) : (f32, f32) -> f32
        %52 = "arith.addf"(%51, %48) : (f32, f32) -> f32
        %53 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
        %h_y = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %54 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %55 = "math.fpowi"(%h_y, %54) : (f32, i64) -> f32
        %56 = "arith.mulf"(%53, %55) : (f32, f32) -> f32
        %57 = "arith.mulf"(%56, %13) : (f32, f32) -> f32
        %58 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
        %h_y_1 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %59 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %60 = "math.fpowi"(%h_y_1, %59) : (f32, i64) -> f32
        %61 = "arith.mulf"(%58, %60) : (f32, f32) -> f32
        %62 = "arith.mulf"(%61, %14) : (f32, f32) -> f32
        %63 = "arith.constant"() {"value" = -2.5 : f32} : () -> f32
        %h_y_2 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %64 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %65 = "math.fpowi"(%h_y_2, %64) : (f32, i64) -> f32
        %66 = "arith.mulf"(%63, %65) : (f32, f32) -> f32
        %67 = "arith.mulf"(%66, %8) : (f32, f32) -> f32
        %68 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
        %h_y_3 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %69 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %70 = "math.fpowi"(%h_y_3, %69) : (f32, i64) -> f32
        %71 = "arith.mulf"(%68, %70) : (f32, f32) -> f32
        %72 = "arith.mulf"(%71, %15) : (f32, f32) -> f32
        %73 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
        %h_y_4 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %74 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %75 = "math.fpowi"(%h_y_4, %74) : (f32, i64) -> f32
        %76 = "arith.mulf"(%73, %75) : (f32, f32) -> f32
        %77 = "arith.mulf"(%76, %16) : (f32, f32) -> f32
        %78 = "arith.addf"(%57, %62) : (f32, f32) -> f32
        %79 = "arith.addf"(%78, %67) : (f32, f32) -> f32
        %80 = "arith.addf"(%79, %72) : (f32, f32) -> f32
        %81 = "arith.addf"(%80, %77) : (f32, f32) -> f32
        %82 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
        %h_z = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %83 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %84 = "math.fpowi"(%h_z, %83) : (f32, i64) -> f32
        %85 = "arith.mulf"(%82, %84) : (f32, f32) -> f32
        %86 = "arith.mulf"(%85, %17) : (f32, f32) -> f32
        %87 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
        %h_z_1 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %88 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %89 = "math.fpowi"(%h_z_1, %88) : (f32, i64) -> f32
        %90 = "arith.mulf"(%87, %89) : (f32, f32) -> f32
        %91 = "arith.mulf"(%90, %18) : (f32, f32) -> f32
        %92 = "arith.constant"() {"value" = -2.5 : f32} : () -> f32
        %h_z_2 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %93 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %94 = "math.fpowi"(%h_z_2, %93) : (f32, i64) -> f32
        %95 = "arith.mulf"(%92, %94) : (f32, f32) -> f32
        %96 = "arith.mulf"(%95, %8) : (f32, f32) -> f32
        %97 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
        %h_z_3 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %98 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %99 = "math.fpowi"(%h_z_3, %98) : (f32, i64) -> f32
        %100 = "arith.mulf"(%97, %99) : (f32, f32) -> f32
        %101 = "arith.mulf"(%100, %19) : (f32, f32) -> f32
        %102 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
        %h_z_4 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %103 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %104 = "math.fpowi"(%h_z_4, %103) : (f32, i64) -> f32
        %105 = "arith.mulf"(%102, %104) : (f32, f32) -> f32
        %106 = "arith.mulf"(%105, %20) : (f32, f32) -> f32
        %107 = "arith.addf"(%86, %91) : (f32, f32) -> f32
        %108 = "arith.addf"(%107, %96) : (f32, f32) -> f32
        %109 = "arith.addf"(%108, %101) : (f32, f32) -> f32
        %110 = "arith.addf"(%109, %106) : (f32, f32) -> f32
        %111 = "arith.addf"(%52, %81) : (f32, f32) -> f32
        %112 = "arith.addf"(%111, %110) : (f32, f32) -> f32
        %a = "arith.constant"() {"value" = 0.5 : f32} : () -> f32
        %113 = "arith.mulf"(%112, %a) : (f32, f32) -> f32
        %114 = "arith.addf"(%23, %113) : (f32, f32) -> f32
        %dt_1 = "arith.constant"() {"value" = 4.122440608513459e-06 : f32} : () -> f32
        %115 = "arith.mulf"(%114, %dt_1) : (f32, f32) -> f32
        "stencil.return"(%115) : (f32) -> ()
      }) : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f32>) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f32>
      "stencil.store"(%7, %t1_w_size_3) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[100 : i64, 100 : i64, 100 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f32>, !stencil.field<[100 : i64, 100 : i64, 100 : i64], f32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "myfunc", "function_type" = (memref<2xmemref<?x?x?xf32>>) -> (), "sym_visibility" = "private", "param_names" = ["data"]} : () -> ()
}) : () -> ()


// CHECK-NEXT:"builtin.module"() ({
// CHECK-NEXT:  "func.func"() ({
// CHECK-NEXT:  ^0(%data : memref<2xmemref<?x?x?xf32>>):
// CHECK-NEXT:    %time_m = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:    %time_M = "arith.constant"() {"value" = 1000 : index} : () -> index
// CHECK-NEXT:    %step = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:    %0 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:    %time_M_1 = "arith.addi"(%time_M, %0) : (index, index) -> index
// CHECK-NEXT:    "scf.for"(%time_m, %time_M_1, %step) ({
// CHECK-NEXT:    ^1(%time : index):
// CHECK-NEXT:      %time_1 = "arith.index_cast"(%time) : (index) -> i64
// CHECK-NEXT:      %1 = "arith.constant"() {"value" = 2 : i64} : () -> i64
// CHECK-NEXT:      %2 = "arith.constant"() {"value" = 0 : i64} : () -> i64
// CHECK-NEXT:      %3 = "arith.addi"(%time_1, %2) : (i64, i64) -> i64
// CHECK-NEXT:      %t0 = "arith.remsi"(%3, %1) : (i64, i64) -> i64
// CHECK-NEXT:      %4 = "arith.constant"() {"value" = 1 : i64} : () -> i64
// CHECK-NEXT:      %5 = "arith.addi"(%time_1, %4) : (i64, i64) -> i64
// CHECK-NEXT:      %t1 = "arith.remsi"(%5, %1) : (i64, i64) -> i64
// CHECK-NEXT:      %t0_w_size = "arith.index_cast"(%t0) : (i64) -> index
// CHECK-NEXT:      %t0_w_size_1 = "memref.load"(%data, %t0_w_size) : (memref<2xmemref<?x?x?xf32>>, index) -> memref<?x?x?xf32>
// CHECK-NEXT:      %t0_w_size_3 = "memref.cast"(%t0_w_size_1) : (memref<?x?x?xf32>) -> memref<108x108x108xf32>
// CHECK-NEXT:      %t1_w_size = "arith.index_cast"(%t1) : (i64) -> index
// CHECK-NEXT:      %t1_w_size_1 = "memref.load"(%data, %t1_w_size) : (memref<2xmemref<?x?x?xf32>>, index) -> memref<?x?x?xf32>
// CHECK-NEXT:      %t1_w_size_3 = "memref.cast"(%t1_w_size_1) : (memref<?x?x?xf32>) -> memref<108x108x108xf32>
// CHECK-NEXT:      %6 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:      %7 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:      %8 = "arith.constant"() {"value" = 100 : index} : () -> index
// CHECK-NEXT:      %9 = "arith.constant"() {"value" = 100 : index} : () -> index
// CHECK-NEXT:      %10 = "arith.constant"() {"value" = 100 : index} : () -> index
// CHECK-NEXT:      "scf.parallel"(%6, %6, %6, %8, %9, %10, %7, %7, %7) ({
// CHECK-NEXT:      ^2(%11 : index, %12 : index, %13 : index):
// CHECK-NEXT:        %14 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %15 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %16 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %17 = "arith.addi"(%13, %14) : (index, index) -> index
// CHECK-NEXT:        %18 = "arith.addi"(%12, %15) : (index, index) -> index
// CHECK-NEXT:        %19 = "arith.addi"(%11, %16) : (index, index) -> index
// CHECK-NEXT:        %20 = "memref.load"(%t0_w_size_3, %17, %18, %19) : (memref<108x108x108xf32>, index, index, index) -> f32
// CHECK-NEXT:        %21 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %22 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %23 = "arith.constant"() {"value" = 3 : index} : () -> index
// CHECK-NEXT:        %24 = "arith.addi"(%13, %21) : (index, index) -> index
// CHECK-NEXT:        %25 = "arith.addi"(%12, %22) : (index, index) -> index
// CHECK-NEXT:        %26 = "arith.addi"(%11, %23) : (index, index) -> index
// CHECK-NEXT:        %27 = "memref.load"(%t0_w_size_3, %24, %25, %26) : (memref<108x108x108xf32>, index, index, index) -> f32
// CHECK-NEXT:        %28 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %29 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %30 = "arith.constant"() {"value" = 5 : index} : () -> index
// CHECK-NEXT:        %31 = "arith.addi"(%13, %28) : (index, index) -> index
// CHECK-NEXT:        %32 = "arith.addi"(%12, %29) : (index, index) -> index
// CHECK-NEXT:        %33 = "arith.addi"(%11, %30) : (index, index) -> index
// CHECK-NEXT:        %34 = "memref.load"(%t0_w_size_3, %31, %32, %33) : (memref<108x108x108xf32>, index, index, index) -> f32
// CHECK-NEXT:        %35 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %36 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %37 = "arith.constant"() {"value" = 2 : index} : () -> index
// CHECK-NEXT:        %38 = "arith.addi"(%13, %35) : (index, index) -> index
// CHECK-NEXT:        %39 = "arith.addi"(%12, %36) : (index, index) -> index
// CHECK-NEXT:        %40 = "arith.addi"(%11, %37) : (index, index) -> index
// CHECK-NEXT:        %41 = "memref.load"(%t0_w_size_3, %38, %39, %40) : (memref<108x108x108xf32>, index, index, index) -> f32
// CHECK-NEXT:        %42 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %43 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %44 = "arith.constant"() {"value" = 6 : index} : () -> index
// CHECK-NEXT:        %45 = "arith.addi"(%13, %42) : (index, index) -> index
// CHECK-NEXT:        %46 = "arith.addi"(%12, %43) : (index, index) -> index
// CHECK-NEXT:        %47 = "arith.addi"(%11, %44) : (index, index) -> index
// CHECK-NEXT:        %48 = "memref.load"(%t0_w_size_3, %45, %46, %47) : (memref<108x108x108xf32>, index, index, index) -> f32
// CHECK-NEXT:        %49 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %50 = "arith.constant"() {"value" = 3 : index} : () -> index
// CHECK-NEXT:        %51 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %52 = "arith.addi"(%13, %49) : (index, index) -> index
// CHECK-NEXT:        %53 = "arith.addi"(%12, %50) : (index, index) -> index
// CHECK-NEXT:        %54 = "arith.addi"(%11, %51) : (index, index) -> index
// CHECK-NEXT:        %55 = "memref.load"(%t0_w_size_3, %52, %53, %54) : (memref<108x108x108xf32>, index, index, index) -> f32
// CHECK-NEXT:        %56 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %57 = "arith.constant"() {"value" = 5 : index} : () -> index
// CHECK-NEXT:        %58 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %59 = "arith.addi"(%13, %56) : (index, index) -> index
// CHECK-NEXT:        %60 = "arith.addi"(%12, %57) : (index, index) -> index
// CHECK-NEXT:        %61 = "arith.addi"(%11, %58) : (index, index) -> index
// CHECK-NEXT:        %62 = "memref.load"(%t0_w_size_3, %59, %60, %61) : (memref<108x108x108xf32>, index, index, index) -> f32
// CHECK-NEXT:        %63 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %64 = "arith.constant"() {"value" = 2 : index} : () -> index
// CHECK-NEXT:        %65 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %66 = "arith.addi"(%13, %63) : (index, index) -> index
// CHECK-NEXT:        %67 = "arith.addi"(%12, %64) : (index, index) -> index
// CHECK-NEXT:        %68 = "arith.addi"(%11, %65) : (index, index) -> index
// CHECK-NEXT:        %69 = "memref.load"(%t0_w_size_3, %66, %67, %68) : (memref<108x108x108xf32>, index, index, index) -> f32
// CHECK-NEXT:        %70 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %71 = "arith.constant"() {"value" = 6 : index} : () -> index
// CHECK-NEXT:        %72 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %73 = "arith.addi"(%13, %70) : (index, index) -> index
// CHECK-NEXT:        %74 = "arith.addi"(%12, %71) : (index, index) -> index
// CHECK-NEXT:        %75 = "arith.addi"(%11, %72) : (index, index) -> index
// CHECK-NEXT:        %76 = "memref.load"(%t0_w_size_3, %73, %74, %75) : (memref<108x108x108xf32>, index, index, index) -> f32
// CHECK-NEXT:        %77 = "arith.constant"() {"value" = 3 : index} : () -> index
// CHECK-NEXT:        %78 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %79 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %80 = "arith.addi"(%13, %77) : (index, index) -> index
// CHECK-NEXT:        %81 = "arith.addi"(%12, %78) : (index, index) -> index
// CHECK-NEXT:        %82 = "arith.addi"(%11, %79) : (index, index) -> index
// CHECK-NEXT:        %83 = "memref.load"(%t0_w_size_3, %80, %81, %82) : (memref<108x108x108xf32>, index, index, index) -> f32
// CHECK-NEXT:        %84 = "arith.constant"() {"value" = 5 : index} : () -> index
// CHECK-NEXT:        %85 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %86 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %87 = "arith.addi"(%13, %84) : (index, index) -> index
// CHECK-NEXT:        %88 = "arith.addi"(%12, %85) : (index, index) -> index
// CHECK-NEXT:        %89 = "arith.addi"(%11, %86) : (index, index) -> index
// CHECK-NEXT:        %90 = "memref.load"(%t0_w_size_3, %87, %88, %89) : (memref<108x108x108xf32>, index, index, index) -> f32
// CHECK-NEXT:        %91 = "arith.constant"() {"value" = 2 : index} : () -> index
// CHECK-NEXT:        %92 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %93 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %94 = "arith.addi"(%13, %91) : (index, index) -> index
// CHECK-NEXT:        %95 = "arith.addi"(%12, %92) : (index, index) -> index
// CHECK-NEXT:        %96 = "arith.addi"(%11, %93) : (index, index) -> index
// CHECK-NEXT:        %97 = "memref.load"(%t0_w_size_3, %94, %95, %96) : (memref<108x108x108xf32>, index, index, index) -> f32
// CHECK-NEXT:        %98 = "arith.constant"() {"value" = 6 : index} : () -> index
// CHECK-NEXT:        %99 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %100 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %101 = "arith.addi"(%13, %98) : (index, index) -> index
// CHECK-NEXT:        %102 = "arith.addi"(%12, %99) : (index, index) -> index
// CHECK-NEXT:        %103 = "arith.addi"(%11, %100) : (index, index) -> index
// CHECK-NEXT:        %104 = "memref.load"(%t0_w_size_3, %101, %102, %103) : (memref<108x108x108xf32>, index, index, index) -> f32
// CHECK-NEXT:        %dt = "arith.constant"() {"value" = 4.122440608513459e-06 : f32} : () -> f32
// CHECK-NEXT:        %105 = "arith.constant"() {"value" = -1 : i64} : () -> i64
// CHECK-NEXT:        %106 = "math.fpowi"(%dt, %105) : (f32, i64) -> f32
// CHECK-NEXT:        %107 = "arith.mulf"(%106, %20) : (f32, f32) -> f32
// CHECK-NEXT:        %108 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
// CHECK-NEXT:        %h_x = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %109 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %110 = "math.fpowi"(%h_x, %109) : (f32, i64) -> f32
// CHECK-NEXT:        %111 = "arith.mulf"(%108, %110) : (f32, f32) -> f32
// CHECK-NEXT:        %112 = "arith.mulf"(%111, %27) : (f32, f32) -> f32
// CHECK-NEXT:        %113 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
// CHECK-NEXT:        %h_x_1 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %114 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %115 = "math.fpowi"(%h_x_1, %114) : (f32, i64) -> f32
// CHECK-NEXT:        %116 = "arith.mulf"(%113, %115) : (f32, f32) -> f32
// CHECK-NEXT:        %117 = "arith.mulf"(%116, %34) : (f32, f32) -> f32
// CHECK-NEXT:        %118 = "arith.constant"() {"value" = -2.5 : f32} : () -> f32
// CHECK-NEXT:        %h_x_2 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %119 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %120 = "math.fpowi"(%h_x_2, %119) : (f32, i64) -> f32
// CHECK-NEXT:        %121 = "arith.mulf"(%118, %120) : (f32, f32) -> f32
// CHECK-NEXT:        %122 = "arith.mulf"(%121, %20) : (f32, f32) -> f32
// CHECK-NEXT:        %123 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
// CHECK-NEXT:        %h_x_3 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %124 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %125 = "math.fpowi"(%h_x_3, %124) : (f32, i64) -> f32
// CHECK-NEXT:        %126 = "arith.mulf"(%123, %125) : (f32, f32) -> f32
// CHECK-NEXT:        %127 = "arith.mulf"(%126, %41) : (f32, f32) -> f32
// CHECK-NEXT:        %128 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
// CHECK-NEXT:        %h_x_4 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %129 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %130 = "math.fpowi"(%h_x_4, %129) : (f32, i64) -> f32
// CHECK-NEXT:        %131 = "arith.mulf"(%128, %130) : (f32, f32) -> f32
// CHECK-NEXT:        %132 = "arith.mulf"(%131, %48) : (f32, f32) -> f32
// CHECK-NEXT:        %133 = "arith.addf"(%112, %117) : (f32, f32) -> f32
// CHECK-NEXT:        %134 = "arith.addf"(%133, %122) : (f32, f32) -> f32
// CHECK-NEXT:        %135 = "arith.addf"(%134, %127) : (f32, f32) -> f32
// CHECK-NEXT:        %136 = "arith.addf"(%135, %132) : (f32, f32) -> f32
// CHECK-NEXT:        %137 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
// CHECK-NEXT:        %h_y = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %138 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %139 = "math.fpowi"(%h_y, %138) : (f32, i64) -> f32
// CHECK-NEXT:        %140 = "arith.mulf"(%137, %139) : (f32, f32) -> f32
// CHECK-NEXT:        %141 = "arith.mulf"(%140, %55) : (f32, f32) -> f32
// CHECK-NEXT:        %142 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
// CHECK-NEXT:        %h_y_1 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %143 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %144 = "math.fpowi"(%h_y_1, %143) : (f32, i64) -> f32
// CHECK-NEXT:        %145 = "arith.mulf"(%142, %144) : (f32, f32) -> f32
// CHECK-NEXT:        %146 = "arith.mulf"(%145, %62) : (f32, f32) -> f32
// CHECK-NEXT:        %147 = "arith.constant"() {"value" = -2.5 : f32} : () -> f32
// CHECK-NEXT:        %h_y_2 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %148 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %149 = "math.fpowi"(%h_y_2, %148) : (f32, i64) -> f32
// CHECK-NEXT:        %150 = "arith.mulf"(%147, %149) : (f32, f32) -> f32
// CHECK-NEXT:        %151 = "arith.mulf"(%150, %20) : (f32, f32) -> f32
// CHECK-NEXT:        %152 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
// CHECK-NEXT:        %h_y_3 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %153 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %154 = "math.fpowi"(%h_y_3, %153) : (f32, i64) -> f32
// CHECK-NEXT:        %155 = "arith.mulf"(%152, %154) : (f32, f32) -> f32
// CHECK-NEXT:        %156 = "arith.mulf"(%155, %69) : (f32, f32) -> f32
// CHECK-NEXT:        %157 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
// CHECK-NEXT:        %h_y_4 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %158 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %159 = "math.fpowi"(%h_y_4, %158) : (f32, i64) -> f32
// CHECK-NEXT:        %160 = "arith.mulf"(%157, %159) : (f32, f32) -> f32
// CHECK-NEXT:        %161 = "arith.mulf"(%160, %76) : (f32, f32) -> f32
// CHECK-NEXT:        %162 = "arith.addf"(%141, %146) : (f32, f32) -> f32
// CHECK-NEXT:        %163 = "arith.addf"(%162, %151) : (f32, f32) -> f32
// CHECK-NEXT:        %164 = "arith.addf"(%163, %156) : (f32, f32) -> f32
// CHECK-NEXT:        %165 = "arith.addf"(%164, %161) : (f32, f32) -> f32
// CHECK-NEXT:        %166 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
// CHECK-NEXT:        %h_z = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %167 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %168 = "math.fpowi"(%h_z, %167) : (f32, i64) -> f32
// CHECK-NEXT:        %169 = "arith.mulf"(%166, %168) : (f32, f32) -> f32
// CHECK-NEXT:        %170 = "arith.mulf"(%169, %83) : (f32, f32) -> f32
// CHECK-NEXT:        %171 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
// CHECK-NEXT:        %h_z_1 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %172 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %173 = "math.fpowi"(%h_z_1, %172) : (f32, i64) -> f32
// CHECK-NEXT:        %174 = "arith.mulf"(%171, %173) : (f32, f32) -> f32
// CHECK-NEXT:        %175 = "arith.mulf"(%174, %90) : (f32, f32) -> f32
// CHECK-NEXT:        %176 = "arith.constant"() {"value" = -2.5 : f32} : () -> f32
// CHECK-NEXT:        %h_z_2 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %177 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %178 = "math.fpowi"(%h_z_2, %177) : (f32, i64) -> f32
// CHECK-NEXT:        %179 = "arith.mulf"(%176, %178) : (f32, f32) -> f32
// CHECK-NEXT:        %180 = "arith.mulf"(%179, %20) : (f32, f32) -> f32
// CHECK-NEXT:        %181 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
// CHECK-NEXT:        %h_z_3 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %182 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %183 = "math.fpowi"(%h_z_3, %182) : (f32, i64) -> f32
// CHECK-NEXT:        %184 = "arith.mulf"(%181, %183) : (f32, f32) -> f32
// CHECK-NEXT:        %185 = "arith.mulf"(%184, %97) : (f32, f32) -> f32
// CHECK-NEXT:        %186 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
// CHECK-NEXT:        %h_z_4 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %187 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %188 = "math.fpowi"(%h_z_4, %187) : (f32, i64) -> f32
// CHECK-NEXT:        %189 = "arith.mulf"(%186, %188) : (f32, f32) -> f32
// CHECK-NEXT:        %190 = "arith.mulf"(%189, %104) : (f32, f32) -> f32
// CHECK-NEXT:        %191 = "arith.addf"(%170, %175) : (f32, f32) -> f32
// CHECK-NEXT:        %192 = "arith.addf"(%191, %180) : (f32, f32) -> f32
// CHECK-NEXT:        %193 = "arith.addf"(%192, %185) : (f32, f32) -> f32
// CHECK-NEXT:        %194 = "arith.addf"(%193, %190) : (f32, f32) -> f32
// CHECK-NEXT:        %195 = "arith.addf"(%136, %165) : (f32, f32) -> f32
// CHECK-NEXT:        %196 = "arith.addf"(%195, %194) : (f32, f32) -> f32
// CHECK-NEXT:        %a = "arith.constant"() {"value" = 0.5 : f32} : () -> f32
// CHECK-NEXT:        %197 = "arith.mulf"(%196, %a) : (f32, f32) -> f32
// CHECK-NEXT:        %198 = "arith.addf"(%107, %197) : (f32, f32) -> f32
// CHECK-NEXT:        %dt_1 = "arith.constant"() {"value" = 4.122440608513459e-06 : f32} : () -> f32
// CHECK-NEXT:        %199 = "arith.mulf"(%198, %dt_1) : (f32, f32) -> f32
// CHECK-NEXT:        %200 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %201 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %202 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:        %203 = "arith.addi"(%13, %200) : (index, index) -> index
// CHECK-NEXT:        %204 = "arith.addi"(%12, %201) : (index, index) -> index
// CHECK-NEXT:        %205 = "arith.addi"(%11, %202) : (index, index) -> index
// CHECK-NEXT:        "memref.store"(%199, %t1_w_size_3, %203, %204, %205) : (f32, memref<108x108x108xf32>, index, index, index) -> ()
// CHECK-NEXT:        "scf.yield"() : () -> ()
// CHECK-NEXT:      }) {"operand_segment_sizes" = array<i32: 3, 3, 3, 0>} : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:      "scf.yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index) -> ()
// CHECK-NEXT:    "func.return"() : () -> ()
// CHECK-NEXT:  }) {"sym_name" = "myfunc", "function_type" = (memref<2xmemref<?x?x?xf32>>) -> (), "sym_visibility" = "private", "param_names" = ["data"]} : () -> ()
// CHECK-NEXT:}) : () -> ()