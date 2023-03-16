// RUN: xdsl-opt %s -t mlir -p convert-stencil-to-ll-mlir | filecheck %s

"builtin.module"() ({
    "func.func"() ({
    ^0(%0 : !stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>, %6 : !stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>):
        %1 = "stencil.cast"(%0) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
        %7 = "stencil.cast"(%6) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
        %2 = "stencil.load"(%1) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> !stencil.temp<[72 : i64, 72 : i64, 72 : i64], f64>
        %8 = "stencil.apply"(%2) ({
        ^b0(%4: !stencil.temp<[72 : i64, 72 : i64, 72 : i64], f64>):
            %5 = "stencil.access"(%4) {"offset" = #stencil.index<[-1 : i64, 0 : i64, 1 : i64]>} : (!stencil.temp<[72 : i64, 72 : i64, 72 : i64], f64>) -> f64
            "stencil.return"(%5) : (f64) -> ()
        }) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.temp<[72 : i64, 72 : i64, 72 : i64], f64>) -> (!stencil.temp<[72 : i64, 72 : i64, 72 : i64], f64>)
        "stencil.store"(%8, %7) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.temp<[72 : i64, 72 : i64, 72 : i64], f64>, !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> ()
        "func.return"() : () -> ()
    }) {"sym_name" = "test_funcop_lowering", "function_type" = (!stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>, !stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK-NEXT: "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : memref<?x?x?xf64>, %1 : memref<?x?x?xf64>):
// CHECK-NEXT:     %2 = "memref.cast"(%0) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %3 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %4 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %5 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %6 = "arith.constant"() {"value" = 72 : index} : () -> index
// CHECK-NEXT:     %7 = "arith.constant"() {"value" = 72 : index} : () -> index
// CHECK-NEXT:     %8 = "arith.constant"() {"value" = 72 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%4, %4, %4, %6, %7, %8, %5, %5, %5) ({
// CHECK-NEXT:     ^1(%9 : index, %10 : index, %11 : index):
// CHECK-NEXT:       %12 = "arith.constant"() {"value" = 3 : index} : () -> index
// CHECK-NEXT:       %13 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %14 = "arith.constant"() {"value" = 5 : index} : () -> index
// CHECK-NEXT:       %15 = "arith.addi"(%9, %12) : (index, index) -> index
// CHECK-NEXT:       %16 = "arith.addi"(%10, %13) : (index, index) -> index
// CHECK-NEXT:       %17 = "arith.addi"(%11, %14) : (index, index) -> index
// CHECK-NEXT:       %18 = "memref.load"(%2, %15, %16, %17) : (memref<72x72x72xf64>, index, index, index) -> f64
// CHECK-NEXT:       %19 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %20 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %21 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %22 = "arith.addi"(%9, %19) : (index, index) -> index
// CHECK-NEXT:       %23 = "arith.addi"(%10, %20) : (index, index) -> index
// CHECK-NEXT:       %24 = "arith.addi"(%11, %21) : (index, index) -> index
// CHECK-NEXT:       "memref.store"(%18, %3, %22, %23, %24) : (f64, memref<72x72x72xf64>, index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 3, 3, 3, 0>} : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"sym_name" = "test_funcop_lowering", "function_type" = (memref<?x?x?xf64>, memref<?x?x?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
// CHECK-NEXT: }) : () -> ()
// CHECK-NEXT: 
