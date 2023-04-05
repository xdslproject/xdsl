// RUN: xdsl-opt %s | filecheck %s

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
      %t0_w_size_3 = "stencil.cast"(%t0_w_size_2) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[54 : i64, 84 : i64, 44 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f32>) -> !stencil.field<[58 : i64, 88 : i64, 48 : i64], f32>
      %t1_w_size = "arith.index_cast"(%t1) : (i64) -> index
      %t1_w_size_1 = "memref.load"(%data, %t1_w_size) : (memref<2xmemref<?x?x?xf32>>, index) -> memref<?x?x?xf32>
      %t1_w_size_2 = "stencil.external_load"(%t1_w_size_1) : (memref<?x?x?xf32>) -> !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f32>
      %t1_w_size_3 = "stencil.cast"(%t1_w_size_2) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[54 : i64, 84 : i64, 44 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f32>) -> !stencil.field<[58 : i64, 88 : i64, 48 : i64], f32>
      %6 = "stencil.load"(%t0_w_size_3) : (!stencil.field<[50 : i64, 80 : i64, 40 : i64], f32>) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f32>
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
      "stencil.store"(%7, %t1_w_size_3) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[50 : i64, 80 : i64, 40 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f32>, !stencil.field<[58 : i64, 88 : i64, 48: i64], f32>) -> ()
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
// CHECK-NEXT:      %t0_w_size_2 = "stencil.external_load"(%t0_w_size_1) : (memref<?x?x?xf32>) -> !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f32>
// CHECK-NEXT:      %t0_w_size_3 = "stencil.cast"(%t0_w_size_2) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[54 : i64, 84 : i64, 44 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f32>) -> !stencil.field<[58 : i64, 88 : i64, 48 : i64], f32>
// CHECK-NEXT:      %t1_w_size = "arith.index_cast"(%t1) : (i64) -> index
// CHECK-NEXT:      %t1_w_size_1 = "memref.load"(%data, %t1_w_size) : (memref<2xmemref<?x?x?xf32>>, index) -> memref<?x?x?xf32>
// CHECK-NEXT:      %t1_w_size_2 = "stencil.external_load"(%t1_w_size_1) : (memref<?x?x?xf32>) -> !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f32>
// CHECK-NEXT:      %t1_w_size_3 = "stencil.cast"(%t1_w_size_2) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[54 : i64, 84 : i64, 44 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f32>) -> !stencil.field<[58 : i64, 88 : i64, 48 : i64], f32>
// CHECK-NEXT:      %6 = "stencil.load"(%t0_w_size_3) : (!stencil.field<[58 : i64, 88 : i64, 48 : i64], f32>) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f32>
// CHECK-NEXT:      %7 = "stencil.apply"(%6) ({
// CHECK-NEXT:      ^2(%t0_buff : !stencil.temp<[-1 : i64], f32>):
// CHECK-NEXT:        %8 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
// CHECK-NEXT:        %9 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[-1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
// CHECK-NEXT:        %10 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
// CHECK-NEXT:        %11 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[-2 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
// CHECK-NEXT:        %12 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[2 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
// CHECK-NEXT:        %13 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, -1 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
// CHECK-NEXT:        %14 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, 1 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
// CHECK-NEXT:        %15 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, -2 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
// CHECK-NEXT:        %16 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, 2 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
// CHECK-NEXT:        %17 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, 0 : i64, -1 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
// CHECK-NEXT:        %18 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, 0 : i64, 1 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
// CHECK-NEXT:        %19 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, 0 : i64, -2 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
// CHECK-NEXT:        %20 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, 0 : i64, 2 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
// CHECK-NEXT:        %dt = "arith.constant"() {"value" = 4.122440608513459e-06 : f32} : () -> f32
// CHECK-NEXT:        %21 = "arith.constant"() {"value" = -1 : i64} : () -> i64
// CHECK-NEXT:        %22 = "math.fpowi"(%dt, %21) : (f32, i64) -> f32
// CHECK-NEXT:        %23 = "arith.mulf"(%22, %8) : (f32, f32) -> f32
// CHECK-NEXT:        %24 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
// CHECK-NEXT:        %h_x = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %25 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %26 = "math.fpowi"(%h_x, %25) : (f32, i64) -> f32
// CHECK-NEXT:        %27 = "arith.mulf"(%24, %26) : (f32, f32) -> f32
// CHECK-NEXT:        %28 = "arith.mulf"(%27, %9) : (f32, f32) -> f32
// CHECK-NEXT:        %29 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
// CHECK-NEXT:        %h_x_1 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %30 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %31 = "math.fpowi"(%h_x_1, %30) : (f32, i64) -> f32
// CHECK-NEXT:        %32 = "arith.mulf"(%29, %31) : (f32, f32) -> f32
// CHECK-NEXT:        %33 = "arith.mulf"(%32, %10) : (f32, f32) -> f32
// CHECK-NEXT:        %34 = "arith.constant"() {"value" = -2.5 : f32} : () -> f32
// CHECK-NEXT:        %h_x_2 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %35 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %36 = "math.fpowi"(%h_x_2, %35) : (f32, i64) -> f32
// CHECK-NEXT:        %37 = "arith.mulf"(%34, %36) : (f32, f32) -> f32
// CHECK-NEXT:        %38 = "arith.mulf"(%37, %8) : (f32, f32) -> f32
// CHECK-NEXT:        %39 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
// CHECK-NEXT:        %h_x_3 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %40 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %41 = "math.fpowi"(%h_x_3, %40) : (f32, i64) -> f32
// CHECK-NEXT:        %42 = "arith.mulf"(%39, %41) : (f32, f32) -> f32
// CHECK-NEXT:        %43 = "arith.mulf"(%42, %11) : (f32, f32) -> f32
// CHECK-NEXT:        %44 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
// CHECK-NEXT:        %h_x_4 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %45 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %46 = "math.fpowi"(%h_x_4, %45) : (f32, i64) -> f32
// CHECK-NEXT:        %47 = "arith.mulf"(%44, %46) : (f32, f32) -> f32
// CHECK-NEXT:        %48 = "arith.mulf"(%47, %12) : (f32, f32) -> f32
// CHECK-NEXT:        %49 = "arith.addf"(%28, %33) : (f32, f32) -> f32
// CHECK-NEXT:        %50 = "arith.addf"(%49, %38) : (f32, f32) -> f32
// CHECK-NEXT:        %51 = "arith.addf"(%50, %43) : (f32, f32) -> f32
// CHECK-NEXT:        %52 = "arith.addf"(%51, %48) : (f32, f32) -> f32
// CHECK-NEXT:        %53 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
// CHECK-NEXT:        %h_y = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %54 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %55 = "math.fpowi"(%h_y, %54) : (f32, i64) -> f32
// CHECK-NEXT:        %56 = "arith.mulf"(%53, %55) : (f32, f32) -> f32
// CHECK-NEXT:        %57 = "arith.mulf"(%56, %13) : (f32, f32) -> f32
// CHECK-NEXT:        %58 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
// CHECK-NEXT:        %h_y_1 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %59 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %60 = "math.fpowi"(%h_y_1, %59) : (f32, i64) -> f32
// CHECK-NEXT:        %61 = "arith.mulf"(%58, %60) : (f32, f32) -> f32
// CHECK-NEXT:        %62 = "arith.mulf"(%61, %14) : (f32, f32) -> f32
// CHECK-NEXT:        %63 = "arith.constant"() {"value" = -2.5 : f32} : () -> f32
// CHECK-NEXT:        %h_y_2 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %64 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %65 = "math.fpowi"(%h_y_2, %64) : (f32, i64) -> f32
// CHECK-NEXT:        %66 = "arith.mulf"(%63, %65) : (f32, f32) -> f32
// CHECK-NEXT:        %67 = "arith.mulf"(%66, %8) : (f32, f32) -> f32
// CHECK-NEXT:        %68 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
// CHECK-NEXT:        %h_y_3 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %69 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %70 = "math.fpowi"(%h_y_3, %69) : (f32, i64) -> f32
// CHECK-NEXT:        %71 = "arith.mulf"(%68, %70) : (f32, f32) -> f32
// CHECK-NEXT:        %72 = "arith.mulf"(%71, %15) : (f32, f32) -> f32
// CHECK-NEXT:        %73 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
// CHECK-NEXT:        %h_y_4 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %74 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %75 = "math.fpowi"(%h_y_4, %74) : (f32, i64) -> f32
// CHECK-NEXT:        %76 = "arith.mulf"(%73, %75) : (f32, f32) -> f32
// CHECK-NEXT:        %77 = "arith.mulf"(%76, %16) : (f32, f32) -> f32
// CHECK-NEXT:        %78 = "arith.addf"(%57, %62) : (f32, f32) -> f32
// CHECK-NEXT:        %79 = "arith.addf"(%78, %67) : (f32, f32) -> f32
// CHECK-NEXT:        %80 = "arith.addf"(%79, %72) : (f32, f32) -> f32
// CHECK-NEXT:        %81 = "arith.addf"(%80, %77) : (f32, f32) -> f32
// CHECK-NEXT:        %82 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
// CHECK-NEXT:        %h_z = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %83 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %84 = "math.fpowi"(%h_z, %83) : (f32, i64) -> f32
// CHECK-NEXT:        %85 = "arith.mulf"(%82, %84) : (f32, f32) -> f32
// CHECK-NEXT:        %86 = "arith.mulf"(%85, %17) : (f32, f32) -> f32
// CHECK-NEXT:        %87 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
// CHECK-NEXT:        %h_z_1 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %88 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %89 = "math.fpowi"(%h_z_1, %88) : (f32, i64) -> f32
// CHECK-NEXT:        %90 = "arith.mulf"(%87, %89) : (f32, f32) -> f32
// CHECK-NEXT:        %91 = "arith.mulf"(%90, %18) : (f32, f32) -> f32
// CHECK-NEXT:        %92 = "arith.constant"() {"value" = -2.5 : f32} : () -> f32
// CHECK-NEXT:        %h_z_2 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %93 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %94 = "math.fpowi"(%h_z_2, %93) : (f32, i64) -> f32
// CHECK-NEXT:        %95 = "arith.mulf"(%92, %94) : (f32, f32) -> f32
// CHECK-NEXT:        %96 = "arith.mulf"(%95, %8) : (f32, f32) -> f32
// CHECK-NEXT:        %97 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
// CHECK-NEXT:        %h_z_3 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %98 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %99 = "math.fpowi"(%h_z_3, %98) : (f32, i64) -> f32
// CHECK-NEXT:        %100 = "arith.mulf"(%97, %99) : (f32, f32) -> f32
// CHECK-NEXT:        %101 = "arith.mulf"(%100, %19) : (f32, f32) -> f32
// CHECK-NEXT:        %102 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
// CHECK-NEXT:        %h_z_4 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
// CHECK-NEXT:        %103 = "arith.constant"() {"value" = -2 : i64} : () -> i64
// CHECK-NEXT:        %104 = "math.fpowi"(%h_z_4, %103) : (f32, i64) -> f32
// CHECK-NEXT:        %105 = "arith.mulf"(%102, %104) : (f32, f32) -> f32
// CHECK-NEXT:        %106 = "arith.mulf"(%105, %20) : (f32, f32) -> f32
// CHECK-NEXT:        %107 = "arith.addf"(%86, %91) : (f32, f32) -> f32
// CHECK-NEXT:        %108 = "arith.addf"(%107, %96) : (f32, f32) -> f32
// CHECK-NEXT:        %109 = "arith.addf"(%108, %101) : (f32, f32) -> f32
// CHECK-NEXT:        %110 = "arith.addf"(%109, %106) : (f32, f32) -> f32
// CHECK-NEXT:        %111 = "arith.addf"(%52, %81) : (f32, f32) -> f32
// CHECK-NEXT:        %112 = "arith.addf"(%111, %110) : (f32, f32) -> f32
// CHECK-NEXT:        %a = "arith.constant"() {"value" = 0.5 : f32} : () -> f32
// CHECK-NEXT:        %113 = "arith.mulf"(%112, %a) : (f32, f32) -> f32
// CHECK-NEXT:        %114 = "arith.addf"(%23, %113) : (f32, f32) -> f32
// CHECK-NEXT:        %dt_1 = "arith.constant"() {"value" = 4.122440608513459e-06 : f32} : () -> f32
// CHECK-NEXT:        %115 = "arith.mulf"(%114, %dt_1) : (f32, f32) -> f32
// CHECK-NEXT:        "stencil.return"(%115) : (f32) -> ()
// CHECK-NEXT:      }) : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f32>) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f32>
// CHECK-NEXT:      "stencil.store"(%7, %t1_w_size_3) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[50 : i64, 80 : i64, 40 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f32>, !stencil.field<[58 : i64, 88 : i64, 48 : i64], f32>) -> ()
// CHECK-NEXT:      "scf.yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index) -> ()
// CHECK-NEXT:    "func.return"() : () -> ()
// CHECK-NEXT:  }) {"sym_name" = "myfunc", "function_type" = (memref<2xmemref<?x?x?xf32>>) -> (), "sym_visibility" = "private", "param_names" = ["data"]} : () -> ()
// CHECK-NEXT:}) : () -> ()
// CHECK-NEXT:
