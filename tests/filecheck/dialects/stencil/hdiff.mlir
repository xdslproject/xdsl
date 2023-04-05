// RUN: xdsl-opt %s -t mlir -p stencil-shape-inference,convert-stencil-to-ll-mlir | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, %1 : !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>):
    %3 = "stencil.cast"(%0) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
    %4 = "stencil.cast"(%1) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
    %6 = "stencil.load"(%3) : (!stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>
    %8 = "stencil.apply"(%6) ({
    ^1(%9 : !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>):
      %10 = "stencil.access"(%9) {"offset" = #stencil.index<[-1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %11 = "stencil.access"(%9) {"offset" = #stencil.index<[1 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %12 = "stencil.access"(%9) {"offset" = #stencil.index<[0 : i64, 1 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %13 = "stencil.access"(%9) {"offset" = #stencil.index<[0 : i64, -1 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %14 = "stencil.access"(%9) {"offset" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> f64
      %15 = "arith.addf"(%10, %11) : (f64, f64) -> f64
      %16 = "arith.addf"(%12, %13) : (f64, f64) -> f64
      %17 = "arith.addf"(%15, %16) : (f64, f64) -> f64
      %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
      %18 = "arith.mulf"(%14, %cst) : (f64, f64) -> f64
      %19 = "arith.addf"(%18, %17) : (f64, f64) -> f64
      "stencil.return"(%19) : (!stencil.result<f64>) -> ()
    }) : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>
    "stencil.store"(%8, %4) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0: i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 64 : i64]>} : (!stencil.temp<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> ()
    "func.return"() : () -> ()
  }) {"function_type" = (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>, !stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> (), "sym_name" = "stencil_hdiff"} : () -> ()
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : memref<?x?x?xf64>, %1 : memref<?x?x?xf64>):
// CHECK-NEXT:     %2 = "memref.cast"(%0) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %3 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %4 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %5 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %6 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %7 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %8 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%4, %4, %4, %6, %7, %8, %5, %5, %5) ({
// CHECK-NEXT:     ^1(%9 : index, %10 : index, %11 : index):
// CHECK-NEXT:       %12 = "arith.constant"() {"value" = 3 : index} : () -> index
// CHECK-NEXT:       %13 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %14 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %15 = "arith.addi"(%9, %12) : (index, index) -> index
// CHECK-NEXT:       %16 = "arith.addi"(%10, %13) : (index, index) -> index
// CHECK-NEXT:       %17 = "arith.addi"(%11, %14) : (index, index) -> index
// CHECK-NEXT:       %18 = "memref.load"(%2, %15, %16, %17) : (memref<72x72x72xf64>, index, index, index) -> f64
// CHECK-NEXT:       %19 = "arith.constant"() {"value" = 5 : index} : () -> index
// CHECK-NEXT:       %20 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %21 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %22 = "arith.addi"(%9, %19) : (index, index) -> index
// CHECK-NEXT:       %23 = "arith.addi"(%10, %20) : (index, index) -> index
// CHECK-NEXT:       %24 = "arith.addi"(%11, %21) : (index, index) -> index
// CHECK-NEXT:       %25 = "memref.load"(%2, %22, %23, %24) : (memref<72x72x72xf64>, index, index, index) -> f64
// CHECK-NEXT:       %26 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %27 = "arith.constant"() {"value" = 5 : index} : () -> index
// CHECK-NEXT:       %28 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %29 = "arith.addi"(%9, %26) : (index, index) -> index
// CHECK-NEXT:       %30 = "arith.addi"(%10, %27) : (index, index) -> index
// CHECK-NEXT:       %31 = "arith.addi"(%11, %28) : (index, index) -> index
// CHECK-NEXT:       %32 = "memref.load"(%2, %29, %30, %31) : (memref<72x72x72xf64>, index, index, index) -> f64
// CHECK-NEXT:       %33 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %34 = "arith.constant"() {"value" = 3 : index} : () -> index
// CHECK-NEXT:       %35 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %36 = "arith.addi"(%9, %33) : (index, index) -> index
// CHECK-NEXT:       %37 = "arith.addi"(%10, %34) : (index, index) -> index
// CHECK-NEXT:       %38 = "arith.addi"(%11, %35) : (index, index) -> index
// CHECK-NEXT:       %39 = "memref.load"(%2, %36, %37, %38) : (memref<72x72x72xf64>, index, index, index) -> f64
// CHECK-NEXT:       %40 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %41 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %42 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %43 = "arith.addi"(%9, %40) : (index, index) -> index
// CHECK-NEXT:       %44 = "arith.addi"(%10, %41) : (index, index) -> index
// CHECK-NEXT:       %45 = "arith.addi"(%11, %42) : (index, index) -> index
// CHECK-NEXT:       %46 = "memref.load"(%2, %43, %44, %45) : (memref<72x72x72xf64>, index, index, index) -> f64
// CHECK-NEXT:       %47 = "arith.addf"(%18, %25) : (f64, f64) -> f64
// CHECK-NEXT:       %48 = "arith.addf"(%32, %39) : (f64, f64) -> f64
// CHECK-NEXT:       %49 = "arith.addf"(%47, %48) : (f64, f64) -> f64
// CHECK-NEXT:       %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
// CHECK-NEXT:       %50 = "arith.mulf"(%46, %cst) : (f64, f64) -> f64
// CHECK-NEXT:       %51 = "arith.addf"(%50, %49) : (f64, f64) -> f64
// CHECK-NEXT:       %52 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %53 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %54 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %55 = "arith.addi"(%9, %52) : (index, index) -> index
// CHECK-NEXT:       %56 = "arith.addi"(%10, %53) : (index, index) -> index
// CHECK-NEXT:       %57 = "arith.addi"(%11, %54) : (index, index) -> index
// CHECK-NEXT:       "memref.store"(%51, %3, %55, %56, %57) : (f64, memref<72x72x72xf64>, index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 3, 3, 3, 0>} : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"function_type" = (memref<?x?x?xf64>, memref<?x?x?xf64>) -> (), "sym_name" = "stencil_hdiff"} : () -> ()
// CHECK-NEXT: }) : () -> ()
