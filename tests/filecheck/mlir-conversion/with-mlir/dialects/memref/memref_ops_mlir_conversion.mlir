// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  "memref.global"() {"alignment" = 64 : i64, "sym_name" = "g_with_alignment", "type" = memref<1xindex>, "initial_value" = dense<0> : tensor<1xindex>, "sym_visibility" = "public"} : () -> ()
  "memref.global"() {"sym_name" = "g", "type" = memref<1xindex>, "initial_value" = dense<0> : tensor<1xindex>, "sym_visibility" = "public"} : () -> ()
  "memref.global"() {"sym_name" = "g_constant", "type" = memref<1xindex>, "initial_value" = dense<0> : tensor<1xindex>, "sym_visibility" = "public", "constant"} : () -> ()
  "func.func"() ({
    %0 = "memref.get_global"() {"name" = @g} : () -> memref<1xindex>
    %1 = "arith.constant"() {"value" = 0 : index} : () -> index
    %2 = "memref.alloca"() {"alignment" = 0 : i64, operandSegmentSizes = array<i32: 0, 0>} : () -> memref<1xindex>
    %3 = "arith.constant"() {"value" = 42 : index} : () -> index
    "memref.store"(%3, %2, %1) : (index, memref<1xindex>, index) -> ()
    %4 = "memref.load"(%2, %1) : (memref<1xindex>, index) -> index
    %5 = "memref.alloc"() {"alignment" = 0 : i64, operandSegmentSizes = array<i32: 0, 0>} : () -> memref<10x2xindex>
    "memref.store"(%3, %5, %3, %4) : (index, memref<10x2xindex>, index, index) -> ()
    %6 = memref.subview %5[0, 0] [1, 1] [1, 1] : memref<10x2xindex> to memref<1x1xindex, strided<[2, 1]>>
    %7 = "memref.cast"(%5) : (memref<10x2xindex>) -> memref<?x?xindex>
    %8 = memref.reinterpret_cast %5 to offset: [0], sizes: [5, 4], strides: [1, 1] : memref<10x2xindex> to memref<5x4xindex, strided<[1, 1]>>
    %no_align = "memref.alloca"() {i64, operandSegmentSizes = array<i32: 0, 0>} : () -> memref<1xindex>
    "memref.copy"(%no_align, %2) : (memref<1xindex>, memref<1xindex>) -> ()
    "memref.dealloc"(%no_align) : (memref<1xindex>) -> ()
    "memref.dealloc"(%2) : (memref<1xindex>) -> ()
    "memref.dealloc"(%5) : (memref<10x2xindex>) -> ()
    %m1 = "memref.alloc"() {"alignment" = 0 : i64, operandSegmentSizes = array<i32: 0, 0>}: () -> memref<100xi32, 10>
    %m2 = "memref.alloc"() {"alignment" = 0 : i64, operandSegmentSizes = array<i32: 0, 0>}: () -> memref<100xi32, 9>
    %tag = "memref.alloc"(){"alignment" = 0 : i64, operandSegmentSizes = array<i32: 0, 0>} : () -> memref<100xi32>
    "memref.dma_start"(%m1, %1, %m2, %1, %3, %tag, %1) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1>} : (
        memref<100xi32, 10>, index,
        memref<100xi32, 9>, index,
        index,
        memref<100xi32>, index
    ) -> ()
    "memref.dma_wait"(%tag, %1, %3) {operandSegmentSizes = array<i32: 1, 1, 1>} : (
        memref<100xi32>, index,
        index
    ) -> ()
    %fmemref = memref.alloc() : memref<32x32xf32>
    %e = arith.constant 1.0 : f32
    %207 = "memref.atomic_rmw"(%e, %fmemref, %1, %1) <{kind = 0 : i64}> : (f32, memref<32x32xf32>, index, index) -> f32
    %index = arith.constant 2 : index
    %dyn_subview = memref.subview %5[%index, 0] [1, 2] [1, 1] : memref<10x2xindex> to memref<2xindex, strided<[1], offset: ?>>
    "func.return"() : () -> ()
  }) {"sym_name" = "memref_test", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()


// CHECK: "builtin.module"() ({
// CHECK-NEXT: "memref.global"() <{alignment = 64 : i64, initial_value = dense<0> : tensor<1xindex>, sym_name = "g_with_alignment", sym_visibility = "public", type = memref<1xindex>}> : () -> ()
// CHECK-NEXT: "memref.global"() <{initial_value = dense<0> : tensor<1xindex>, sym_name = "g", sym_visibility = "public", type = memref<1xindex>}> : () -> ()
// CHECK-NEXT: "memref.global"() <{constant, initial_value = dense<0> : tensor<1xindex>, sym_name = "g_constant", sym_visibility = "public", type = memref<1xindex>}> : () -> ()
// CHECK-NEXT: "func.func"() <{function_type = () -> (), sym_name = "memref_test", sym_visibility = "private"}> ({
// CHECK-NEXT: %0 = "memref.get_global"() <{name = @g}> : () -> memref<1xindex>
// CHECK-NEXT: %1 = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT: %2 = "memref.alloca"() <{alignment = 0 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xindex>
// CHECK-NEXT: %3 = "arith.constant"() <{value = 42 : index}> : () -> index
// CHECK-NEXT: "memref.store"(%3, %2, %1) : (index, memref<1xindex>, index) -> ()
// CHECK-NEXT: %4 = "memref.load"(%2, %1) : (memref<1xindex>, index) -> index
// CHECK-NEXT: %5 = "memref.alloc"() <{alignment = 0 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x2xindex>
// CHECK-NEXT: "memref.store"(%3, %5, %3, %4) : (index, memref<10x2xindex>, index, index) -> ()
// CHECK-NEXT: %6 = "memref.subview"(%5) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1>, static_strides = array<i64: 1, 1>}> : (memref<10x2xindex>) -> memref<1x1xindex, strided<[2, 1]>>
// CHECK-NEXT: %7 = "memref.cast"(%5) : (memref<10x2xindex>) -> memref<?x?xindex>
// CHECK-NEXT: %8 = "memref.reinterpret_cast"(%5) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 5, 4>, static_strides = array<i64: 1, 1>}> : (memref<10x2xindex>) -> memref<5x4xindex, strided<[1, 1]>>
// CHECK-NEXT: %9 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> {i64} : () -> memref<1xindex>
// CHECK-NEXT: "memref.copy"(%9, %2) : (memref<1xindex>, memref<1xindex>) -> ()
// CHECK-NEXT: "memref.dealloc"(%9) : (memref<1xindex>) -> ()
// CHECK-NEXT: "memref.dealloc"(%2) : (memref<1xindex>) -> ()
// CHECK-NEXT: "memref.dealloc"(%5) : (memref<10x2xindex>) -> ()

// CHECK:      "memref.dma_start"(%10, %1, %11, %1, %3, %12, %1) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1>} : (memref<100xi32, 10 : i64>, index, memref<100xi32, 9 : i64>, index, index, memref<100xi32>, index) -> ()
// CHECK-NEXT: "memref.dma_wait"(%12, %1, %3) {operandSegmentSizes = array<i32: 1, 1, 1>} : (memref<100xi32>, index, index) -> ()
// CHECK-NEXT: %13 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32>
// CHECK-NEXT: %14 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
// CHECK-NEXT: %15 = "memref.atomic_rmw"(%14, %13, %1, %1) <{kind = 0 : i64}> : (f32, memref<32x32xf32>, index, index) -> f32
// CHECK-NEXT: %16 = "arith.constant"() <{value = 2 : index}> : () -> index
// CHECK-NEXT: %17 = "memref.subview"(%5, %16) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 1, 2>, static_strides = array<i64: 1, 1>}> : (memref<10x2xindex>, index) -> memref<2xindex, strided<[1], offset: ?>>
// CHECK-NEXT: "func.return"() : () -> ()
// CHECK-NEXT: }) : () -> ()
// CHECK-NEXT: }) : () -> ()
