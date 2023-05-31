// RUN: xdsl-opt %s -p stencil-shape-inference,convert-stencil-to-ll-mlir | filecheck %s

builtin.module {
  func.func @bufferswapping_stencil(%f0 : !stencil.field<[-2,2002]x[-2,2002]xf32>, %f1 : !stencil.field<[-2,2002]x[-2,2002]xf32>) -> (!stencil.field<[-2,2002]x[-2,2002]xf32>) {

    %time_m = "arith.constant"() {"value" = 0 : index} : () -> index
    %time_M = "arith.constant"() {"value" = 1001 : index} : () -> index
    %step = "arith.constant"() {"value" = 1 : index} : () -> index

    %t1_out, %t0_out = "scf.for"(%time_m, %time_M, %step, %f0, %f1) ({
    ^1(%time : index, %fim1 : !stencil.field<[-2,2002]x[-2,2002]xf32>, %fi : !stencil.field<[-2,2002]x[-2,2002]xf32>):

      %tim1 = "stencil.load"(%fim1) : (!stencil.field<[-2,2002]x[-2,2002]xf32>) -> !stencil.temp<?x?xf32>

      %ti = "stencil.apply"(%tim1) ({
      ^2(%tim1_b : !stencil.temp<?x?xf32>):
        %i = "stencil.access"(%tim1) {"offset" = #stencil.index<0,0,0>} : (!stencil.temp<?x?xf32>) -> f32
        "stencil.return"(%i) : (f32) -> ()
      }) : (!stencil.temp<?x?xf32>) -> !stencil.temp<?x?xf32>

      "stencil.store"(%ti, %fi) {"lb" = #stencil.index<0, 0>, "ub" = #stencil.index<2000, 2000>} : (!stencil.temp<?x?xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>) -> ()

      "scf.yield"(%fi, %fim1) : (!stencil.field<[-2,2002]x[-2,2002]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>) -> ()
    }) : (index, index, index, !stencil.field<[-2,2002]x[-2,2002]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>) -> (!stencil.field<[-2,2002]x[-2,2002]xf32>, !stencil.field<[-2,2002]x[-2,2002]xf32>)

    "func.return"(%t1_out) : (!stencil.field<[-2,2002]x[-2,2002]xf32>) -> ()
  }
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func @bufferswapping_stencil(%f0 : memref<2004x2004xf32>, %f1 : memref<2004x2004xf32>) -> memref<2004x2004xf32> {
// CHECK-NEXT:     %time_m = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:     %time_M = "arith.constant"() {"value" = 1001 : index} : () -> index
// CHECK-NEXT:     %step = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:     %t1_out, %t0_out = "scf.for"(%time_m, %time_M, %step, %f0, %f1) ({
// CHECK-NEXT:     ^0(%time : index, %fim1 : memref<2004x2004xf32>, %fi : memref<2004x2004xf32>):
// CHECK-NEXT:       %fi_1 = "memref.subview"(%fi) {"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 2000, 2000>, "static_strides" = array<i64: 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<2004x2004xf32>) -> memref<2004x2004xf32>
// CHECK-NEXT:       %tim1 = "memref.subview"(%fim1) {"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 2000, 2000>, "static_strides" = array<i64: 1, 1>, "operand_segment_sizes" = array<i32: 1, 0, 0, 0>} : (memref<2004x2004xf32>) -> memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>
// CHECK-NEXT:       %0 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:       %1 = "arith.constant"() {"value" = 1 : index} : () -> index
// CHECK-NEXT:       %2 = "arith.constant"() {"value" = 2000 : index} : () -> index
// CHECK-NEXT:       %3 = "arith.constant"() {"value" = 2000 : index} : () -> index
// CHECK-NEXT:       "scf.parallel"(%0, %2, %1) ({
// CHECK-NEXT:       ^1(%4 : index):
// CHECK-NEXT:         "scf.for"(%0, %3, %1) ({
// CHECK-NEXT:         ^2(%5 : index):
// CHECK-NEXT:           %i = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:           %i_1 = "arith.constant"() {"value" = 0 : index} : () -> index
// CHECK-NEXT:           %i_2 = arith.addi %4, %i : index
// CHECK-NEXT:           %i_3 = arith.addi %5, %i_1 : index
// CHECK-NEXT:           %i_4 = "memref.load"(%tim1, %i_2, %i_3) : (memref<2000x2000xf32, strided<[2004, 1], offset: 4010>>, index, index) -> f32
// CHECK-NEXT:           "memref.store"(%i_4, %fi_1, %4, %5) : (f32, memref<2004x2004xf32>, index, index) -> ()
// CHECK-NEXT:           "scf.yield"() : () -> ()
// CHECK-NEXT:         }) : (index, index, index) -> ()
// CHECK-NEXT:         "scf.yield"() : () -> ()
// CHECK-NEXT:       }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 0>} : (index, index, index) -> ()
// CHECK-NEXT:       "scf.yield"(%fi_1, %fim1) : (memref<2004x2004xf32>, memref<2004x2004xf32>) -> ()
// CHECK-NEXT:     }) : (index, index, index, memref<2004x2004xf32>, memref<2004x2004xf32>) -> (memref<2004x2004xf32>, memref<2004x2004xf32>)
// CHECK-NEXT:     "func.return"(%t1_out) : (memref<2004x2004xf32>) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }