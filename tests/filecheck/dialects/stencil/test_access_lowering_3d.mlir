// RUN: xdsl-opt %s -p convert-stencil-to-ll-mlir --print-op-generic | filecheck %s

"builtin.module"() ({
    "func.func"() ({
    ^0(%0 : !stencil.field<?x?x?xf64>):
        %1 = "stencil.cast"(%0) : (!stencil.field<?x?x?xf64>) -> !stencil.field<[-4,68]x[-4,70]x[-4,72]xf64>
        %2 = "stencil.load"(%1) : (!stencil.field<[-4,68]x[-4,70]x[-4,72]xf64>) -> !stencil.temp<[-4,68]x[-4,70]x[-4,72]xf64>
        %5 = "stencil.apply"(%2) ({
        ^b0(%3: !stencil.temp<[-4,68]x[-4,70]x[-4,72]xf64>):
            %4 = "stencil.access"(%3) {"offset" = #stencil.index<-1, 0, 1>} : (!stencil.temp<[-4,68]x[-4,70]x[-4,72]xf64>) -> f64
            "stencil.return"(%4) : (f64) -> ()
        }) : (!stencil.temp<[-4,68]x[-4,70]x[-4,72]xf64>) -> (!stencil.temp<[0,64]x[0,64]x[0,68]xf64>)
        "func.return"() : () -> ()
    }) {"sym_name" = "test_funcop_lowering", "function_type" = (!stencil.field<?x?x?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() ({
// CHECK-NEXT:   ^0(%0 : memref<?x?x?xf64>):
// CHECK-NEXT:     %1 = "memref.cast"(%0) : (memref<?x?x?xf64>) -> memref<72x74x76xf64>
// CHECK-NEXT:     %2 = "memref.subview"(%1) {"static_offsets" = array<i64: 0, 0, 0>, "static_sizes" = array<i64: 72, 74, 76>, "static_strides" = array<i64: 1, 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<72x74x76xf64>) -> memref<72x74x76xf64, strided<[5624, 76, 1]>>
// CHECK-NEXT:     %3 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %4 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %5 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %6 = "arith.constant"() {"value" = 64 : index} : () -> index
// CHECK-NEXT:     %7 = "arith.constant"() {"value" = 68 : index} : () -> index
// CHECK-NEXT:     "scf.parallel"(%3, %5, %4) ({
// CHECK-NEXT:     ^1(%8 : index):
// CHECK-NEXT:       "scf.for"(%3, %6, %4) ({
// CHECK-NEXT:       ^2(%9 : index):
// CHECK-NEXT:         "scf.for"(%3, %7, %4) ({
// CHECK-NEXT:         ^3(%10 : index):
// CHECK-NEXT:           %11 = "arith.constant"() {"value" = 3 : index} : () -> index
// CHECK-NEXT:           %12 = "arith.constant"() {"value" = 4 : index} : () -> index
// CHECK-NEXT:           %13 = "arith.constant"() {"value" = 5 : index} : () -> index
// CHECK-NEXT:           %14 = "arith.addi"(%8, %11) : (index, index) -> index
// CHECK-NEXT:           %15 = "arith.addi"(%9, %12) : (index, index) -> index
// CHECK-NEXT:           %16 = "arith.addi"(%10, %13) : (index, index) -> index
// CHECK-NEXT:           %17 = "memref.load"(%2, %14, %15, %16) : (memref<72x74x76xf64, strided<[5624, 76, 1]>>, index, index, index) -> f64
// CHECK-NEXT:           "scf.yield"() : () -> ()
// CHECK-NEXT:         }) : (index, index, index) -> ()
// CHECK-NEXT:         "scf.yield"() : () -> ()
// CHECK-NEXT:       }) : (index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"() : () -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) {"sym_name" = "test_funcop_lowering", "function_type" = (memref<?x?x?xf64>) -> (), "sym_visibility" = "private"} : () -> ()
// CHECK-NEXT: }) : () -> ()
