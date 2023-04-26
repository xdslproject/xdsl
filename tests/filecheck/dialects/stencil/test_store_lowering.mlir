// RUN: xdsl-opt %s -t mlir -p stencil-shape-inference,convert-stencil-to-ll-mlir | filecheck %s

"builtin.module"() ({
    "func.func"() ({
    ^0(%0 : !stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>, %6 : !stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>):
        %1 = "stencil.cast"(%0) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
        %7 = "stencil.cast"(%6) {"lb" = #stencil.index<[-4 : i64, -4 : i64, -4 : i64]>, "ub" = #stencil.index<[68 : i64, 68 : i64, 68 : i64]>} : (!stencil.field<[-1 : i64, -1 : i64, -1 : i64], f64>) -> !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>
        %2 = "stencil.load"(%1) : (!stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> !stencil.temp<[72 : i64, 72 : i64, 72 : i64], f64>
        %8 = "stencil.apply"(%2) ({
        ^b0(%4: !stencil.temp<[72 : i64, 72 : i64, 72 : i64], f64>):
            %5 = "stencil.access"(%4) {"offset" = #stencil.index<[-1 : i64, 0 : i64, 1 : i64]>} : (!stencil.temp<[72 : i64, 72 : i64, 72 : i64], f64>) -> f64
            "stencil.return"(%5) : (f64) -> ()
        }) : (!stencil.temp<[72 : i64, 72 : i64, 72 : i64], f64>) -> (!stencil.temp<[72 : i64, 72 : i64, 72 : i64], f64>)
        "stencil.store"(%8, %7) {"lb" = #stencil.index<[0 : i64, 0 : i64, 0 : i64]>, "ub" = #stencil.index<[64 : i64, 64 : i64, 64 : i64]>} : (!stencil.temp<[72 : i64, 72 : i64, 72 : i64], f64>, !stencil.field<[72 : i64, 72 : i64, 72 : i64], f64>) -> ()
        "func.return"() : () -> ()
    }) {"sym_name" = "test_funcop_lowering", "function_type" = (!stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>, !stencil.field<[-1 : i32, -1 : i32, -1 : i32], f64>) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : memref<?x?x?xf64>, %1 : memref<?x?x?xf64>):
// CHECK-NEXT:     %2 = "memref.cast"(%0) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %3 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
// CHECK-NEXT:     %4 = "memref.subview"(%2) {"static_offsets" = array<i64: 3, 4, 5>, "static_sizes" = array<i64: 64, 64, 64>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x72x72xf64>) -> memref<64x64x64xf64, strided<[5184, 72, 1], offset: 15845>>
// CHECK-NEXT:     %5 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %6 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %7 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %8 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %9 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%5, %5, %5, %7, %8, %9, %6, %6, %6) ({
// CHECK-NEXT:     ^1(%10 : index, %11 : index, %12 : index):
// CHECK-NEXT:       %13 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %14 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %15 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %16 = "arith.addi"(%10, %13) : (index, index) -> index
// CHECK-NEXT:       %17 = "arith.addi"(%11, %14) : (index, index) -> index
// CHECK-NEXT:       %18 = "arith.addi"(%12, %15) : (index, index) -> index
// CHECK-NEXT:       %19 = "memref.load"(%4, %16, %17, %18) : (memref<64x64x64xf64, strided<[5184, 72, 1], offset: 15845>>, index, index, index) -> f64
// CHECK-NEXT:       %20 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %21 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %22 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:       %23 = "arith.addi"(%10, %20) : (index, index) -> index
// CHECK-NEXT:       %24 = "arith.addi"(%11, %21) : (index, index) -> index
// CHECK-NEXT:       %25 = "arith.addi"(%12, %22) : (index, index) -> index
// CHECK-NEXT:       "memref.store"(%19, %3, %23, %24, %25) : (f64, memref<72x72x72xf64>, index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 3, 3, 3, 0>} : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"sym_name" = "test_funcop_lowering", "function_type" = (memref<?x?x?xf64>, memref<?x?x?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
// CHECK-NEXT: }) : () -> ()
