"builtin.module"() ({
  "func.func"() ({
  ^0(%a : f32, %h_y : f32, %h_x : f32, %dt : f32, %data : memref<2xmemref<?x?xf32>>, %time_M : index, %time_m : index):
    %0 = "arith.constant"() {"value" = 1 : index} : () -> index
    "scf.for"(%time_m, %time_M, %0) ({
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
      %t1_w_size = "arith.index_cast"(%t1) : (i64) -> index

      %t0_w_size_1 = "memref.load"(%data, %t0_w_size) : (memref<2xmemref<?x?xf32>>, index) -> memref<?x?xf32>
      %t1_w_size_1 = "memref.load"(%data, %t1_w_size) : (memref<2xmemref<?x?xf32>>, index) -> memref<?x?xf32>

      %t0_w_size_2 = "stencil.external_load"(%t0_w_size_1) : (memref<?x?xf32>) -> !stencil.field<[-1 : i64, -1 : i64], f32>
      %t1_w_size_2 = "stencil.external_load"(%t1_w_size_1) : (memref<?x?xf32>) -> !stencil.field<[-1 : i64, -1 : i64], f32>

      %t0_w_size_3 = "stencil.cast"(%t0_w_size_2) {"lb" = #stencil.index<[-2 : i64, -2 : i64]>, "ub" = #stencil.index<[130 : i64, 130 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64], f32>) -> !stencil.field<[132 : i64, 132 : i64], f32>
      %t1_w_size_3 = "stencil.cast"(%t1_w_size_2) {"lb" = #stencil.index<[-2 : i64, -2 : i64]>, "ub" = #stencil.index<[130 : i64, 130 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64], f32>) -> !stencil.field<[132 : i64, 132 : i64], f32>

      %input = "stencil.load"(%t0_w_size_3) : (!stencil.field<[132 : i64, 132 : i64], f32>) -> !stencil.temp<[-1 : i64, -1 : i64], f32>

      %result = "stencil.apply"(%input) ({
      ^2(%t0_buff : !stencil.temp<[-1 : i64, -1: i64], f32>):
        %8 = "stencil.access"(%t0_buff)  {"offset" = #stencil.index<[0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %9 = "stencil.access"(%t0_buff)  {"offset" = #stencil.index<[-2 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %10 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[2 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %11 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, -2 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        %12 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<[0 : i64, 2 : i64]>} : (!stencil.temp<[-1 : i64], f32>) -> f32
        "stencil.return"(%8) : (f32) -> ()
      }) : (!stencil.temp<[-1 : i64, -1 : i64], f32>) -> !stencil.temp<[-1 : i64, -1 : i64], f32>

      "stencil.store"(%result, %t1_w_size_3) {"lb" = #stencil.index<[0 : i64, 0 : i64]>, "ub" = #stencil.index<[32 : i64, 32 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64], f32>, !stencil.field<[132 : i64, 132 : i64], f32>) -> ()

      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()

    "func.return"() : () -> ()

  }) {"sym_name" = "kernel", "function_type" = (f32, f32, f32, f32, memref<2xmemref<?x?xf32>>, index, index) -> (), "sym_visibility" = "private", "param_names" = ["a", "h_y", "h_x", "dt", "data", "time_M", "time_m"]} : () -> ()
}) : () -> ()