// RUN: xdsl-opt %s -p convert-stencil-to-ll-mlir --print-op-generic | filecheck %s

"builtin.module"() ({
    "func.func"() ({
    ^0(%0 : !stencil.field<[-1 : i32], f64>):
        %1 = "stencil.cast"(%0) {"lb" = #stencil.index<[-4 : i64]>, "ub" = #stencil.index<[68 : i64]>} : (!stencil.field<[-1 : i64], f64>) -> !stencil.field<[72 : i64], f64>
        %2 = "stencil.load"(%1) {"lb" = #stencil.index<[-4 : i64]>, "ub" = #stencil.index<[68 : i64]>} : (!stencil.field<[72 : i64], f64>) -> !stencil.temp<[72 : i64], f64>
        "stencil.apply"(%2) ({
        ^b0(%3: !stencil.temp<[72 : i64], f64>):
            %4 = "stencil.access"(%3) {"offset" = #stencil.index<[-1 : i64, 0 : i64, 1 : i64]>} : (!stencil.temp<[72 : i64], f64>) -> f64
        }) {"lb" = #stencil.index<[0 : i64]>, "ub" = #stencil.index<[68 : i64]>} : (!stencil.temp<[72 : i64], f64>) -> ()
        "func.return"() : () -> ()
    }) {"sym_name" = "test_funcop_lowering", "function_type" = (!stencil.field<[-1 : i32], f64>) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : memref<?xf64>):
// CHECK-NEXT:     %1 = "memref.cast"(%0) : (memref<?xf64>) -> memref<72xf64>
// CHECK-NEXT:     %2 = "memref.subview"(%1) {"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 72>, "static_strides" = array<i64: 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72xf64>) -> memref<72xf64, strided<[1]>>
// CHECK-NEXT:     %3 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %4 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %5 = "arith.constant"() {"value" = 68 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%3, %5, %4) ({
// CHECK-NEXT:     ^1(%6 : index):
// CHECK-NEXT:       %7 = "arith.constant"() {"value" = 3 : index} : () -> index
// CHECK-NEXT:       %8 = "arith.addi"(%6, %7) : (index, index) -> index
// CHECK-NEXT:       %9 = "memref.load"(%2, %8) : (memref<72xf64, strided<[1]>>, index) -> f64
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"sym_name" = "test_funcop_lowering", "function_type" = (memref<?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
// CHECK-NEXT: }) : () -> ()
