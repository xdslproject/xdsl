// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  "memref.global"() {"alignment" = 64 : i64, "sym_name" = "g_with_alignment", "type" = memref<1xindex>, "initial_value" = dense<0> : tensor<1xindex>, "sym_visibility" = "public"} : () -> ()
  "memref.global"() {"sym_name" = "g", "type" = memref<1xindex>, "initial_value" = dense<0> : tensor<1xindex>, "sym_visibility" = "public"} : () -> ()
  "func.func"() ({
    %0 = "memref.get_global"() {"name" = @g} : () -> memref<1xindex>
    %1 = "arith.constant"() {"value" = 0 : index} : () -> index
    %2 = "memref.alloca"() {"alignment" = 0 : i64, "operandSegmentSizes" = array<i32: 0, 0>} : () -> memref<1xindex>
    %3 = "arith.constant"() {"value" = 42 : index} : () -> index
    "memref.store"(%3, %2, %1) : (index, memref<1xindex>, index) -> ()
    %4 = "memref.load"(%2, %1) : (memref<1xindex>, index) -> index
    %5 = "memref.alloc"() {"alignment" = 0 : i64, "operandSegmentSizes" = array<i32: 0, 0>} : () -> memref<10x2xindex>
    "memref.store"(%3, %5, %3, %4) : (index, memref<10x2xindex>, index, index) -> ()
    %6 = "memref.subview"(%5) {"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 0, 0>, "static_sizes" = array<i64: 1, 1>, "static_strides" = array<i64: 1, 1>} : (memref<10x2xindex>) -> memref<1x1xindex>
    %7 = "memref.cast"(%5) : (memref<10x2xindex>) -> memref<?x?xindex>
    %no_align = "memref.alloca"() {i64, "operandSegmentSizes" = array<i32: 0, 0>} : () -> memref<1xindex>
    "memref.copy"(%no_align, %2) : (memref<1xindex>, memref<1xindex>) -> ()
    "memref.dealloc"(%no_align) : (memref<1xindex>) -> ()
    "memref.dealloc"(%2) : (memref<1xindex>) -> ()
    "memref.dealloc"(%5) : (memref<10x2xindex>) -> ()
    %m1 = "memref.alloc"() {"alignment" = 0 : i64, "operandSegmentSizes" = array<i32: 0, 0>}: () -> memref<100xi32, 10>
    %m2 = "memref.alloc"() {"alignment" = 0 : i64, "operandSegmentSizes" = array<i32: 0, 0>}: () -> memref<100xi32, 9>
    %tag = "memref.alloc"(){"alignment" = 0 : i64, "operandSegmentSizes" = array<i32: 0, 0>} : () -> memref<100xi32>
    "memref.dma_start"(%m1, %1, %m2, %1, %3, %tag, %1) {"operandSegmentSizes" = array<i32: 1, 1, 1, 1, 1, 1, 1>} : (
        memref<100xi32, 10>, index,
        memref<100xi32, 9>, index,
        index,
        memref<100xi32>, index
    ) -> ()
    "memref.dma_wait"(%tag, %1, %3) {"operandSegmentSizes" = array<i32: 1, 1, 1>} : (
        memref<100xi32>, index,
        index
    ) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "memref_test", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()


// CHECK: "builtin.module"() ({
// CHECK-NEXT: "memref.global"() <{"alignment" = 64 : i64, "initial_value" = dense<0> : tensor<1xindex>, "sym_name" = "g_with_alignment", "sym_visibility" = "public", "type" = memref<1xindex>}> : () -> ()
// CHECK-NEXT: "memref.global"() <{"initial_value" = dense<0> : tensor<1xindex>, "sym_name" = "g", "sym_visibility" = "public", "type" = memref<1xindex>}> : () -> ()
// CHECK-NEXT: "func.func"() <{"function_type" = () -> (), "sym_name" = "memref_test", "sym_visibility" = "private"}> ({
// CHECK-NEXT: %0 = "memref.get_global"() <{"name" = @g}> : () -> memref<1xindex>
// CHECK-NEXT: %1 = "arith.constant"() <{"value" = 0 : index}> : () -> index
// CHECK-NEXT: %2 = "memref.alloca"() <{"alignment" = 0 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<1xindex>
// CHECK-NEXT: %3 = "arith.constant"() <{"value" = 42 : index}> : () -> index
// CHECK-NEXT: "memref.store"(%3, %2, %1) : (index, memref<1xindex>, index) -> ()
// CHECK-NEXT: %4 = "memref.load"(%2, %1) : (memref<1xindex>, index) -> index
// CHECK-NEXT: %5 = "memref.alloc"() <{"alignment" = 0 : i64, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<10x2xindex>
// CHECK-NEXT: "memref.store"(%3, %5, %3, %4) : (index, memref<10x2xindex>, index, index) -> ()
// CHECK-NEXT: %6 = "memref.subview"(%5) <{"operandSegmentSizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 0, 0>, "static_sizes" = array<i64: 1, 1>, "static_strides" = array<i64: 1, 1>}> : (memref<10x2xindex>) -> memref<1x1xindex>
// CHECK-NEXT: %7 = "memref.cast"(%5) : (memref<10x2xindex>) -> memref<?x?xindex>
// CHECK-NEXT: %8 = "memref.alloca"() <{"operandSegmentSizes" = array<i32: 0, 0>}> {"i64"} : () -> memref<1xindex>
// CHECK-NEXT: "memref.copy"(%8, %2) : (memref<1xindex>, memref<1xindex>) -> ()
// CHECK-NEXT: "memref.dealloc"(%8) : (memref<1xindex>) -> ()
// CHECK-NEXT: "memref.dealloc"(%2) : (memref<1xindex>) -> ()
// CHECK-NEXT: "memref.dealloc"(%5) : (memref<10x2xindex>) -> ()

// CHECK:      "memref.dma_start"(%9, %1, %10, %1, %3, %11, %1) {"operandSegmentSizes" = array<i32: 1, 1, 1, 1, 1, 1, 1>} : (memref<100xi32, 10 : i64>, index, memref<100xi32, 9 : i64>, index, index, memref<100xi32>, index) -> ()
// CHECK-NEXT: "memref.dma_wait"(%11, %1, %3) {"operandSegmentSizes" = array<i32: 1, 1, 1>} : (memref<100xi32>, index, index) -> ()
// CHECK-NEXT: "func.return"() : () -> ()
// CHECK-NEXT: }) : () -> ()
// CHECK-NEXT: }) : () -> ()
