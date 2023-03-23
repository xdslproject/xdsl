// RUN: xdsl-opt %s -t mlir -p stencil-shape-inference,convert-stencil-to-gpu | filecheck %s


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


// CHECK-NEXT: "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : memref<?x?x?xf64>, %1 : memref<?x?x?xf64>):
// CHECK-NEXT:     %2 = "memref.cast"(%0) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %3 = "memref.cast"(%2) : (memref<72x72x72xf64>) -> memref<*xf64>
// CHECK-NEXT:     "gpu.host_register"(%3) : (memref<*xf64>) -> ()
// CHECK-NEXT:     %4 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %5 = "memref.cast"(%4) : (memref<72x72x72xf64>) -> memref<*xf64>
// CHECK-NEXT:     "gpu.host_register"(%5) : (memref<*xf64>) -> ()
// CHECK-NEXT:     %6 = "arith.constant"() {"value" = 8 : index} : () -> index
// CHECK-NEXT:     %7 = "arith.constant"() {"value" = 8 : index} : () -> index
// CHECK-NEXT:     %8 = "arith.constant"() {"value" = 8 : index} : () -> index
// CHECK-NEXT:     %9 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %10 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %11 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %12 = "arith.ceildivui"(%9, %6) : (index, index) -> index
// CHECK-NEXT:     %13 = "arith.ceildivui"(%10, %7) : (index, index) -> index
// CHECK-NEXT:     %14 = "arith.ceildivui"(%11, %8) : (index, index) -> index
// CHECK-NEXT:     %15 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     "gpu.launch"(%12, %13, %14, %6, %7, %8) ({
// CHECK-NEXT:     ^1(%16 : index, %17 : index, %18 : index, %19 : index, %20 : index, %21 : index, %22 : index, %23 : index, %24 : index, %25 : index, %26 : index, %27 : index):
// CHECK-NEXT:       %28 = "arith.muli"(%16, %25) : (index, index) -> index
// CHECK-NEXT:       %29 = "arith.muli"(%17, %26) : (index, index) -> index
// CHECK-NEXT:       %30 = "arith.muli"(%18, %27) : (index, index) -> index
// CHECK-NEXT:       %31 = "arith.addi"(%19, %28) : (index, index) -> index
// CHECK-NEXT:       %32 = "arith.addi"(%20, %29) : (index, index) -> index
// CHECK-NEXT:       %33 = "arith.addi"(%21, %30) : (index, index) -> index
// CHECK-NEXT:       %34 = "arith.index_cast"(%31) : (index) -> i64
// CHECK-NEXT:       %35 = "arith.index_cast"(%32) : (index) -> i64
// CHECK-NEXT:       %36 = "arith.index_cast"(%33) : (index) -> i64
// CHECK-NEXT:       %37 = "arith.index_cast"(%9) : (index) -> i64
// CHECK-NEXT:       %38 = "arith.index_cast"(%10) : (index) -> i64
// CHECK-NEXT:       %39 = "arith.index_cast"(%11) : (index) -> i64
// CHECK-NEXT:       %40 = "arith.cmpi"(%34, %37) {"predicate" = 6 : i64} : (i64, i64) -> i1
// CHECK-NEXT:       %41 = "arith.cmpi"(%35, %38) {"predicate" = 6 : i64} : (i64, i64) -> i1
// CHECK-NEXT:       %42 = "arith.cmpi"(%36, %39) {"predicate" = 6 : i64} : (i64, i64) -> i1
// CHECK-NEXT:       %43 = "arith.andi"(%40, %41) : (i1, i1) -> i1
// CHECK-NEXT:       %44 = "arith.andi"(%42, %43) : (i1, i1) -> i1
// CHECK-NEXT:       "cf.cond_br"(%44, %31, %32, %33) [^2, ^3] {"operand_segment_sizes" = array<i32: 1, 3, 0>} : (i1, index, index, index) -> ()
// CHECK-NEXT:     ^2(%45 : index, %46 : index, %47 : index):
// CHECK-NEXT:       %48 = "arith.constant"() {"value" = 3 : index} : () -> index
// CHECK-NEXT:       %49 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %50 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %51 = "arith.addi"(%45, %48) : (index, index) -> index
// CHECK-NEXT:       %52 = "arith.addi"(%46, %49) : (index, index) -> index
// CHECK-NEXT:       %53 = "arith.addi"(%47, %50) : (index, index) -> index
// CHECK-NEXT:       %54 = "memref.load"(%2, %51, %52, %53) : (memref<72x72x72xf64>, index, index, index) -> f64
// CHECK-NEXT:       %55 = "arith.constant"() {"value" = 5 : index} : () -> index
// CHECK-NEXT:       %56 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %57 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %58 = "arith.addi"(%45, %55) : (index, index) -> index
// CHECK-NEXT:       %59 = "arith.addi"(%46, %56) : (index, index) -> index
// CHECK-NEXT:       %60 = "arith.addi"(%47, %57) : (index, index) -> index
// CHECK-NEXT:       %61 = "memref.load"(%2, %58, %59, %60) : (memref<72x72x72xf64>, index, index, index) -> f64
// CHECK-NEXT:       %62 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %63 = "arith.constant"() {"value" = 5 : index} : () -> index
// CHECK-NEXT:       %64 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %65 = "arith.addi"(%45, %62) : (index, index) -> index
// CHECK-NEXT:       %66 = "arith.addi"(%46, %63) : (index, index) -> index
// CHECK-NEXT:       %67 = "arith.addi"(%47, %64) : (index, index) -> index
// CHECK-NEXT:       %68 = "memref.load"(%2, %65, %66, %67) : (memref<72x72x72xf64>, index, index, index) -> f64
// CHECK-NEXT:       %69 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %70 = "arith.constant"() {"value" = 3 : index} : () -> index
// CHECK-NEXT:       %71 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %72 = "arith.addi"(%45, %69) : (index, index) -> index
// CHECK-NEXT:       %73 = "arith.addi"(%46, %70) : (index, index) -> index
// CHECK-NEXT:       %74 = "arith.addi"(%47, %71) : (index, index) -> index
// CHECK-NEXT:       %75 = "memref.load"(%2, %72, %73, %74) : (memref<72x72x72xf64>, index, index, index) -> f64
// CHECK-NEXT:       %76 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %77 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %78 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %79 = "arith.addi"(%45, %76) : (index, index) -> index
// CHECK-NEXT:       %80 = "arith.addi"(%46, %77) : (index, index) -> index
// CHECK-NEXT:       %81 = "arith.addi"(%47, %78) : (index, index) -> index
// CHECK-NEXT:       %82 = "memref.load"(%2, %79, %80, %81) : (memref<72x72x72xf64>, index, index, index) -> f64
// CHECK-NEXT:       %83 = "arith.addf"(%54, %61) : (f64, f64) -> f64
// CHECK-NEXT:       %84 = "arith.addf"(%68, %75) : (f64, f64) -> f64
// CHECK-NEXT:       %85 = "arith.addf"(%83, %84) : (f64, f64) -> f64
// CHECK-NEXT:       %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
// CHECK-NEXT:       %86 = "arith.mulf"(%82, %cst) : (f64, f64) -> f64
// CHECK-NEXT:       %87 = "arith.addf"(%86, %85) : (f64, f64) -> f64
// CHECK-NEXT:       %88 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %89 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %90 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %91 = "arith.addi"(%45, %88) : (index, index) -> index
// CHECK-NEXT:       %92 = "arith.addi"(%46, %89) : (index, index) -> index
// CHECK-NEXT:       %93 = "arith.addi"(%47, %90) : (index, index) -> index
// CHECK-NEXT:       "memref.store"(%87, %4, %91, %92, %93) : (f64, memref<72x72x72xf64>, index, index, index) -> ()
// CHECK-NEXT:       "gpu.terminator"() : () -> ()
// CHECK-NEXT:     ^3:
// CHECK-NEXT:       "gpu.terminator"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 0, 1, 1, 1, 1, 1, 1, 0>} : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"function_type" = (memref<?x?x?xf64>, memref<?x?x?xf64>) -> (), "sym_name" = "stencil_hdiff"} : () -> ()
// CHECK-NEXT: }) : () -> ()

