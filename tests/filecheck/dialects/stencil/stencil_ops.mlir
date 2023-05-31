// RUN: xdsl-opt %s --print-op-generic --split-input-file | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>):
    %2 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %3 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
    %4 = "stencil.load"(%2) : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
    %5 = "stencil.apply"(%4) ({
    ^1(%6 : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>):
      %7 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> f64
      %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
      "stencil.return"(%8) : (!stencil.result<f64>) -> ()
    }) : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
    "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "stencil_copy", "function_type" = (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : !stencil.field<?x?x?xf64>, %1 : !stencil.field<?x?x?xf64>):
// CHECK-NEXT:     %2 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %3 = "stencil.cast"(%1) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %4 = "stencil.load"(%2) : (!stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// CHECK-NEXT:     %5 = "stencil.apply"(%4) ({
// CHECK-NEXT:     ^1(%6 : !stencil.temp<[0,64]x[0,64]x[0,64]xf64>):
// CHECK-NEXT:       %7 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> f64
// CHECK-NEXT:       %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
// CHECK-NEXT:       "stencil.return"(%8) : (!stencil.result<f64>) -> ()
// CHECK-NEXT:     }) : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,64]xf64>
// CHECK-NEXT:     "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<64, 64, 64>} : (!stencil.temp<[0,64]x[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]x[-4,68]xf64>) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"sym_name" = "stencil_copy", "function_type" = (!stencil.field<?x?x?xf64>, !stencil.field<?x?x?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
// CHECK-NEXT: }) : () -> ()

// -----

"builtin.module"() ({
  "func.func"() ({
  ^0(%data : memref<2xmemref<?x?x?xf32>>):
    %time_m = "arith.constant"() {"value" = 0 : index} : () -> index
    %time_M = "arith.constant"() {"value" = 1000 : index} : () -> index
    %step = "arith.constant"() {"value" = 1 : index} : () -> index
    "scf.for"(%time_m, %time_M, %step) ({
    ^1(%time : index):
      %time_1 = "arith.index_cast"(%time) : (index) -> i64
      %0 = "arith.constant"() {"value" = 2 : i64} : () -> i64
      %1 = "arith.constant"() {"value" = 0 : i64} : () -> i64
      %2 = "arith.addi"(%time_1, %1) : (i64, i64) -> i64
      %t0 = "arith.remsi"(%2, %0) : (i64, i64) -> i64
      %3 = "arith.constant"() {"value" = 1 : i64} : () -> i64
      %4 = "arith.addi"(%time_1, %3) : (i64, i64) -> i64
      %t1 = "arith.remsi"(%4, %0) : (i64, i64) -> i64
      %t0_w_size = "arith.index_cast"(%t0) : (i64) -> index
      %t0_w_size_1 = "memref.load"(%data, %t0_w_size) : (memref<2xmemref<?x?x?xf32>>, index) -> memref<?x?x?xf32>
      %t0_w_size_2 = "stencil.external_load"(%t0_w_size_1) : (memref<?x?x?xf32>) -> !stencil.field<?x?x?xf32>
      %t0_w_size_3 = "stencil.cast"(%t0_w_size_2) : (!stencil.field<?x?x?xf32>) -> !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>
      %t1_w_size = "arith.index_cast"(%t1) : (i64) -> index
      %t1_w_size_1 = "memref.load"(%data, %t1_w_size) : (memref<2xmemref<?x?x?xf32>>, index) -> memref<?x?x?xf32>
      %t1_w_size_2 = "stencil.external_load"(%t1_w_size_1) : (memref<?x?x?xf32>) -> !stencil.field<?x?x?xf32>
      %t1_w_size_3 = "stencil.cast"(%t1_w_size_2) : (!stencil.field<?x?x?xf32>) -> !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>
      %5 = "stencil.load"(%t0_w_size_3) : (!stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>) -> !stencil.temp<?x?x?xf32>
      %6 = "stencil.apply"(%5) ({
      ^2(%t0_buff : !stencil.temp<?xf32>):
        %7 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<?xf32>) -> f32
        "stencil.return"(%7) : (f32) -> ()
      }) : (!stencil.temp<?x?x?xf32>) -> !stencil.temp<?x?x?xf32>
      "stencil.store"(%6, %t1_w_size_3) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<50, 80, 40>} : (!stencil.temp<?x?x?xf32>, !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "myfunc", "function_type" = (memref<2xmemref<?x?x?xf32>>) -> (), "sym_visibility" = "private", "param_names" = ["data"]} : () -> ()
}) : () -> ()




// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%data : memref<2xmemref<?x?x?xf32>>):
// CHECK-NEXT:     %time_m = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %time_M = "arith.constant"() {"value" = 1000 : index} : () -> index
// CHECK-NEXT:     %step = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     "scf.for"(%time_m, %time_M, %step) ({
// CHECK-NEXT:     ^1(%time : index):
// CHECK-NEXT:       %time_1 = "arith.index_cast"(%time) : (index) -> i64
// CHECK-NEXT:       %0 = "arith.constant"() {"value" = 2 : i64} : () -> i64
// CHECK-NEXT:       %1 = "arith.constant"() {"value" = 0 : i64} : () -> i64
// CHECK-NEXT:       %2 = "arith.addi"(%time_1, %1) : (i64, i64) -> i64
// CHECK-NEXT:       %t0 = "arith.remsi"(%2, %0) : (i64, i64) -> i64
// CHECK-NEXT:       %3 = "arith.constant"() {"value" = 1 : i64} : () -> i64
// CHECK-NEXT:       %4 = "arith.addi"(%time_1, %3) : (i64, i64) -> i64
// CHECK-NEXT:       %t1 = "arith.remsi"(%4, %0) : (i64, i64) -> i64
// CHECK-NEXT:       %t0_w_size = "arith.index_cast"(%t0) : (i64) -> index
// CHECK-NEXT:       %t0_w_size_1 = "memref.load"(%data, %t0_w_size) : (memref<2xmemref<?x?x?xf32>>, index) -> memref<?x?x?xf32>
// CHECK-NEXT:       %t0_w_size_2 = "stencil.external_load"(%t0_w_size_1) : (memref<?x?x?xf32>) -> !stencil.field<?x?x?xf32>
// CHECK-NEXT:       %t0_w_size_3 = "stencil.cast"(%t0_w_size_2) : (!stencil.field<?x?x?xf32>) -> !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>
// CHECK-NEXT:       %t1_w_size = "arith.index_cast"(%t1) : (i64) -> index
// CHECK-NEXT:       %t1_w_size_1 = "memref.load"(%data, %t1_w_size) : (memref<2xmemref<?x?x?xf32>>, index) -> memref<?x?x?xf32>
// CHECK-NEXT:       %t1_w_size_2 = "stencil.external_load"(%t1_w_size_1) : (memref<?x?x?xf32>) -> !stencil.field<?x?x?xf32>
// CHECK-NEXT:       %t1_w_size_3 = "stencil.cast"(%t1_w_size_2) : (!stencil.field<?x?x?xf32>) -> !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>
// CHECK-NEXT:       %5 = "stencil.load"(%t0_w_size_3) : (!stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>) -> !stencil.temp<?x?x?xf32>
// CHECK-NEXT:       %6 = "stencil.apply"(%5) ({
// CHECK-NEXT:       ^2(%t0_buff : !stencil.temp<?xf32>):
// CHECK-NEXT:         %7 = "stencil.access"(%t0_buff) {"offset" = #stencil.index<0, 0, 0>} : (!stencil.temp<?xf32>) -> f32
// CHECK-NEXT:         "stencil.return"(%7) : (f32) -> ()
// CHECK-NEXT:       }) : (!stencil.temp<?x?x?xf32>) -> !stencil.temp<?x?x?xf32>
// CHECK-NEXT:       "stencil.store"(%6, %t1_w_size_3) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<50, 80, 40>} : (!stencil.temp<?x?x?xf32>, !stencil.field<[-4,54]x[-4,84]x[-4,44]xf32>) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) : (index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"sym_name" = "myfunc", "function_type" = (memref<2xmemref<?x?x?xf32>>) -> (), "sym_visibility" = "private", "param_names" = ["data"]} : () -> ()
// CHECK-NEXT: }) : () -> ()

// -----

"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : !stencil.field<?x?xf64>, %1 : !stencil.field<?x?xf64>):
    %2 = "stencil.cast"(%0) : (!stencil.field<?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]xf64>
    %3 = "stencil.cast"(%1) : (!stencil.field<?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]xf64>
    %4 = "stencil.load"(%2) : (!stencil.field<[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,65]x[-1,65]xf64>
    %5 = "stencil.apply"(%4) ({
    ^1(%6 : !stencil.temp<[-1,65]x[-1,65]xf64>):
      %7 = "stencil.access"(%6) {"offset" = #stencil.index<-1, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
      %8 = "stencil.access"(%6) {"offset" = #stencil.index<1, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
      %9 = "stencil.access"(%6) {"offset" = #stencil.index<0, 1>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
      %10 = "stencil.access"(%6) {"offset" = #stencil.index<0, -1>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
      %11 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
      %12 = "arith.addf"(%7, %8) : (f64, f64) -> f64
      %13 = "arith.addf"(%9, %10) : (f64, f64) -> f64
      %14 = "arith.addf"(%12, %13) : (f64, f64) -> f64
      %15 = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
      %16 = "arith.mulf"(%11, %15) : (f64, f64) -> f64
      %17 = "arith.mulf"(%16, %13) : (f64, f64) -> f64
      "stencil.return"(%17) : (!stencil.result<f64>) -> ()
    }) : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> !stencil.temp<[0,64]x[0,64]xf64>
    "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0>, "ub" = #stencil.index<64, 64>} : (!stencil.temp<[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]xf64>) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "stencil_laplace", "function_type" = (!stencil.field<?x?xf64>, !stencil.field<?x?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : !stencil.field<?x?xf64>, %1 : !stencil.field<?x?xf64>):
// CHECK-NEXT:     %2 = "stencil.cast"(%0) : (!stencil.field<?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %3 = "stencil.cast"(%1) : (!stencil.field<?x?xf64>) -> !stencil.field<[-4,68]x[-4,68]xf64>
// CHECK-NEXT:     %4 = "stencil.load"(%2) : (!stencil.field<[-4,68]x[-4,68]xf64>) -> !stencil.temp<[-1,65]x[-1,65]xf64>
// CHECK-NEXT:     %5 = "stencil.apply"(%4) ({
// CHECK-NEXT:     ^1(%6 : !stencil.temp<[-1,65]x[-1,65]xf64>):
// CHECK-NEXT:       %7 = "stencil.access"(%6) {"offset" = #stencil.index<-1, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
// CHECK-NEXT:       %8 = "stencil.access"(%6) {"offset" = #stencil.index<1, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
// CHECK-NEXT:       %9 = "stencil.access"(%6) {"offset" = #stencil.index<0, 1>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
// CHECK-NEXT:       %10 = "stencil.access"(%6) {"offset" = #stencil.index<0, -1>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
// CHECK-NEXT:       %11 = "stencil.access"(%6) {"offset" = #stencil.index<0, 0>} : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> f64
// CHECK-NEXT:       %12 = "arith.addf"(%7, %8) : (f64, f64) -> f64
// CHECK-NEXT:       %13 = "arith.addf"(%9, %10) : (f64, f64) -> f64
// CHECK-NEXT:       %14 = "arith.addf"(%12, %13) : (f64, f64) -> f64
// CHECK-NEXT:       %15 = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
// CHECK-NEXT:       %16 = "arith.mulf"(%11, %15) : (f64, f64) -> f64
// CHECK-NEXT:       %17 = "arith.mulf"(%16, %13) : (f64, f64) -> f64
// CHECK-NEXT:       "stencil.return"(%17) : (f64) -> ()
// CHECK-NEXT:     }) : (!stencil.temp<[-1,65]x[-1,65]xf64>) -> !stencil.temp<[0,64]x[0,64]xf64>
// CHECK-NEXT:     "stencil.store"(%5, %3) {"lb" = #stencil.index<0, 0>, "ub" = #stencil.index<64, 64>} : (!stencil.temp<[0,64]x[0,64]xf64>, !stencil.field<[-4,68]x[-4,68]xf64>) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"sym_name" = "stencil_laplace", "function_type" = (!stencil.field<?x?xf64>, !stencil.field<?x?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
// CHECK-NEXT: }) : () -> ()
