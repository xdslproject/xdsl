"builtin.module"() ({
  "func.func"() ({
    %data   = "memref.alloc"() {"alignment" = 64 : i64, "operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<2xmemref<?x?xf32>>
    %d0     = "memref.alloc"() {"alignment" = 64 : i64, "operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<2004x2004xf32>
    %d1     = "memref.alloc"() {"alignment" = 64 : i64, "operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<2004x2004xf32>

    %time_m = "arith.constant"() {"value" = 0 : index} : () -> index
    %step = "arith.constant"() {"value" = 1 : index} : () -> index

    %d0_ = "memref.cast"(%d0) : (memref<2004x2004xf32>) -> memref<?x?xf32>
    %d1_ = "memref.cast"(%d1) : (memref<2004x2004xf32>) -> memref<?x?xf32>

    "memref.store"(%d0_, %data, %time_m) : (memref<?x?xf32>, memref<2xmemref<?x?xf32>>, index) -> ()
    "memref.store"(%d1_, %data, %step) : (memref<?x?xf32>, memref<2xmemref<?x?xf32>>, index) -> ()

    %time_M = "arith.constant"() {"value" = 1000 : index} : () -> index
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
      %t0_w_size_1 = "memref.load"(%data, %t0_w_size) : (memref<2xmemref<?x?xf32>>, index) -> memref<?x?xf32>
      %t0_w_size_2 = "stencil.external_load"(%t0_w_size_1) : (memref<?x?xf32>) -> !stencil.field<[-1 : i64, -1 : i64], f32>
      %t0_w_size_3 = "stencil.cast"(%t0_w_size_2) {"lb" = #stencil.index<[-2 : i64, -2 : i64]>, "ub" = #stencil.index<[2002 : i64, 2002 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64], f32>) -> !stencil.field<[2000 : i64, 2000 : i64], f32>
      %t1_w_size = "arith.index_cast"(%t1) : (i64) -> index
      %t1_w_size_1 = "memref.load"(%data, %t1_w_size) : (memref<2xmemref<?x?xf32>>, index) -> memref<?x?xf32>
      %t1_w_size_2 = "stencil.external_load"(%t1_w_size_1) : (memref<?x?xf32>) -> !stencil.field<[-1 : i64, -1 : i64], f32>
      %t1_w_size_3 = "stencil.cast"(%t1_w_size_2) {"lb" = #stencil.index<[-2 : i64, -2 : i64]>, "ub" = #stencil.index<[2002 : i64, 2002 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64], f32>) -> !stencil.field<[2000 : i64, 2000 : i64], f32>
      %6 = "stencil.load"(%t0_w_size_3) : (!stencil.field<[2000 : i64, 2000 : i64], f32>) -> !stencil.temp<[-1 : i64, -1 : i64], f32>
      %7 = "stencil.apply"(%6) ({
      ^2(%t0_buff : !stencil.temp<[-1 : i64], f32>):
        %8 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %9 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[-1 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %10 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[1 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %11 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, -1 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %12 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, 1 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %13 = "arith.constant"() {"value" = 0.01 : f32} : () -> f32
        %dt = "arith.constant"() {"value" = 5.005003752501563e-07 : f32} : () -> f32
        %14 = "arith.constant"() {"value" = -1 : i64} : () -> i64
        %15 = "math.fpowi"(%dt, %14) : (f32, i64) -> f32
        %16 = "arith.mulf"(%15, %8) : (f32, f32) -> f32
        %h_x = "arith.constant"() {"value" = 0.0005002501420676708 : f32} : () -> f32
        %17 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %18 = "math.fpowi"(%h_x, %17) : (f32, i64) -> f32
        %19 = "arith.mulf"(%18, %9) : (f32, f32) -> f32
        %h_x_1 = "arith.constant"() {"value" = 0.0005002501420676708 : f32} : () -> f32
        %20 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %21 = "math.fpowi"(%h_x_1, %20) : (f32, i64) -> f32
        %22 = "arith.mulf"(%21, %10) : (f32, f32) -> f32
        %23 = "arith.constant"() {"value" = -2.0 : f32} : () -> f32
        %h_x_2 = "arith.constant"() {"value" = 0.0005002501420676708 : f32} : () -> f32
        %24 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %25 = "math.fpowi"(%h_x_2, %24) : (f32, i64) -> f32
        %26 = "arith.mulf"(%23, %25) : (f32, f32) -> f32
        %27 = "arith.mulf"(%26, %8) : (f32, f32) -> f32
        %28 = "arith.addf"(%19, %22) : (f32, f32) -> f32
        %29 = "arith.addf"(%28, %27) : (f32, f32) -> f32
        %h_y = "arith.constant"() {"value" = 0.0005002501420676708 : f32} : () -> f32
        %30 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %31 = "math.fpowi"(%h_y, %30) : (f32, i64) -> f32
        %32 = "arith.mulf"(%31, %11) : (f32, f32) -> f32
        %h_y_1 = "arith.constant"() {"value" = 0.0005002501420676708 : f32} : () -> f32
        %33 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %34 = "math.fpowi"(%h_y_1, %33) : (f32, i64) -> f32
        %35 = "arith.mulf"(%34, %12) : (f32, f32) -> f32
        %36 = "arith.constant"() {"value" = -2.0 : f32} : () -> f32
        %h_y_2 = "arith.constant"() {"value" = 0.0005002501420676708 : f32} : () -> f32
        %37 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %38 = "math.fpowi"(%h_y_2, %37) : (f32, i64) -> f32
        %39 = "arith.mulf"(%36, %38) : (f32, f32) -> f32
        %40 = "arith.mulf"(%39, %8) : (f32, f32) -> f32
        %41 = "arith.addf"(%32, %35) : (f32, f32) -> f32
        %42 = "arith.addf"(%41, %40) : (f32, f32) -> f32
        %43 = "arith.addf"(%29, %42) : (f32, f32) -> f32
        %a = "arith.constant"() {"value" = 0.1 : f32} : () -> f32
        %44 = "arith.mulf"(%43, %a) : (f32, f32) -> f32
        %45 = "arith.addf"(%13, %16) : (f32, f32) -> f32
        %46 = "arith.addf"(%45, %44) : (f32, f32) -> f32
        %dt_1 = "arith.constant"() {"value" = 5.005003752501563e-07 : f32} : () -> f32
        %47 = "arith.mulf"(%46, %dt_1) : (f32, f32) -> f32
        "stencil.return"(%47) : (f32) -> ()
      }) : (!stencil.temp<[-1 : i64, -1 : i64], f32>) -> !stencil.temp<[-1 : i64, -1 : i64], f32>
      "stencil.store"(%7, %t1_w_size_3) {"lb" = #stencil.index<[0 : i64, 0 : i64]>, "ub" = #stencil.index<[2000 : i64, 2000 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64], f32>, !stencil.field<[2000 : i64, 2000 : i64], f32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "main", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()
