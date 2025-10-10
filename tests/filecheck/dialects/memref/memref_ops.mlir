// RUN: XDSL_ROUNDTRIP

builtin.module {
  func.func @memref_alloca_scope() {
    "memref.alloca_scope"() ({
      "memref.alloca_scope.return"() : () -> ()
    }) : () -> ()
    func.return
  }
  "memref.global"() {"sym_name" = "g", "type" = memref<1xindex>, "initial_value" = dense<0> : tensor<1xindex>, "sym_visibility" = "public"} : () -> ()
  "memref.global"() {"sym_name" = "g_constant", "type" = memref<1xindex>, "initial_value" = dense<0> : tensor<1xindex>, "sym_visibility" = "public", "constant"} : () -> ()
  "memref.global"() {"alignment" = 64 : i64, "sym_name" = "g_with_alignment", "type" = memref<1xindex>, "initial_value" = dense<0> : tensor<1xindex>, "sym_visibility" = "public"} : () -> ()
  func.func private @memref_test() {
    %0 = "memref.get_global"() {"name" = @g} : () -> memref<1xindex>
    %1 = arith.constant 0 : index
    %2 = "memref.alloca"() {"alignment" = 0 : i64, operandSegmentSizes = array<i32: 0, 0>} : () -> memref<1xindex>
    %3 = arith.constant 42 : index
    memref.store %3, %2[%1] : memref<1xindex>
    memref.store %3, %2[%1] {nontemporal = true} : memref<1xindex>
    memref.store %3, %2[%1] {nontemporal = false} : memref<1xindex>
    %4 = memref.load %2[%1] : memref<1xindex>
    %f = memref.load %2[%1] {nontemporal = false} : memref<1xindex>
    %g = memref.load %2[%1] {nontemporal = true} : memref<1xindex>
    %5 = memref.alloc() {"alignment" = 0} : memref<10x2xindex>
    "memref.store"(%3, %5, %3, %4) : (index, memref<10x2xindex>, index, index) -> ()
    %6 = memref.subview %5[0, 0] [1, 1] [1, 1] {"hello" = "world"} : memref<10x2xindex> to memref<1x1xindex>
    %7 = "memref.cast"(%5) : (memref<10x2xindex>) -> memref<?x?xindex>
    %8 = "memref.alloca"() {operandSegmentSizes = array<i32: 0, 0>} : () -> memref<1xindex>
    %9 = "memref.memory_space_cast"(%5) : (memref<10x2xindex>) -> memref<10x2xindex, 1: i32>
    %10 = memref.alloc() : memref<64x64xindex, strided<[2, 4], offset: 6>, 2 : i32>
    %11 = "memref.alloca"() {operandSegmentSizes = array<i32: 0, 0>} : () -> memref<64x64xindex, strided<[2, 4], offset: 6>, 2 : i32>
    %base_buffer, %offset, %sizes:2, %strides:2 = "memref.extract_strided_metadata"(%11) : (memref<64x64xindex, strided<[2, 4], offset: 6>, 2 : i32>) -> (memref<index>, index, index, index, index, index)
    %12, %13, %14 = "test.op"() : () -> (index, index, index)
    %15 = memref.alloc(%12) {"alignment" = 0} : memref<?xindex>
    %16 = memref.alloc(%12, %13, %14) {"alignment" = 0} : memref<?x?x?xindex>
    %17 = "memref.alloca"(%12) {"alignment" = 0 : i64, operandSegmentSizes = array<i32: 1, 0>} : (index) -> memref<?xindex>
    %18 = "memref.alloca"(%12, %13, %14) {"alignment" = 0 : i64, operandSegmentSizes = array<i32: 3, 0>} : (index, index, index) -> memref<?x?x?xindex>
    %19 = memref.collapse_shape %5 [[0, 1]] : memref<10x2xindex> into memref<20xindex>
    %20 = arith.constant 2 : index
    %21 = memref.expand_shape %19 [[0, 1]] output_shape [%20, 10] : memref<20xindex> into memref<?x10xindex>
    %22 = memref.reinterpret_cast %5 to offset: [0], sizes: [5, 4], strides: [1, 1] : memref<10x2xindex> to memref<5x4xindex>
    memref.dealloc %2 : memref<1xindex>
    memref.dealloc %5 : memref<10x2xindex>
    memref.dealloc %8 : memref<1xindex>
    memref.dealloc %10 : memref<64x64xindex, strided<[2, 4], offset: 6>, 2 : i32>
    memref.dealloc %11 : memref<64x64xindex, strided<[2, 4], offset: 6>, 2 : i32>
    %fmemref = "test.op"() : () -> memref<32x32xf32>
    %e = "test.op"() : () -> f32
    %207 = "memref.atomic_rmw"(%e, %fmemref, %1, %1) <{kind = 0 : i64}> : (f32, memref<32x32xf32>, index, index) -> f32

    func.return
  }
}

// CHECK:  builtin.module {
// CHECK-NEXT:    func.func @memref_alloca_scope() {
// CHECK-NEXT:      "memref.alloca_scope"() ({
// CHECK-NEXT:        "memref.alloca_scope.return"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:   "memref.global"() <{sym_name = "g", sym_visibility = "public", type = memref<1xindex>, initial_value = dense<0> : tensor<1xindex>}> : () -> ()
// CHECK-NEXT:   "memref.global"() <{sym_name = "g_constant", sym_visibility = "public", type = memref<1xindex>, initial_value = dense<0> : tensor<1xindex>, constant}> : () -> ()
// CHECK-NEXT:   "memref.global"() <{sym_name = "g_with_alignment", sym_visibility = "public", type = memref<1xindex>, initial_value = dense<0> : tensor<1xindex>, alignment = 64 : i64}> : () -> ()
// CHECK-NEXT:   func.func private @memref_test() {
// CHECK-NEXT:     %{{.*}} = memref.get_global @g : memref<1xindex>
// CHECK-NEXT:     %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:     %{{.*}} = "memref.alloca"() <{alignment = 0 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xindex>
// CHECK-NEXT:     %{{.*}} = arith.constant 42 : index
// CHECK-NEXT:     memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1xindex>
// CHECK-NEXT:     memref.store %{{.*}}, %{{.*}}[%{{.*}}] {nontemporal = true} : memref<1xindex>
// CHECK-NEXT:     memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1xindex>
// CHECK-NEXT:     %{{.*}} = memref.load %{{.*}}[%{{.*}}] : memref<1xindex>
// CHECK-NEXT:     %{{.*}} = memref.load %{{.*}}[%{{.*}}] : memref<1xindex>
// CHECK-NEXT:     %{{.*}} = memref.load %{{.*}}[%{{.*}}] {nontemporal = true} : memref<1xindex>
// CHECK-NEXT:     %{{.*}} = memref.alloc() {alignment = 0 : i64} : memref<10x2xindex>
// CHECK-NEXT:     memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<10x2xindex>
// CHECK-NEXT:     %{{.*}} = memref.subview %{{.*}}[0, 0] [1, 1] [1, 1] {hello = "world"} : memref<10x2xindex> to memref<1x1xindex>
// CHECK-NEXT:     %{{.*}} = "memref.cast"(%{{.*}}) : (memref<10x2xindex>) -> memref<?x?xindex>
// CHECK-NEXT:     %{{.*}} = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xindex>
// CHECK-NEXT:     %{{.*}} = "memref.memory_space_cast"(%{{.*}}) : (memref<10x2xindex>) -> memref<10x2xindex, 1 : i32>
// CHECK-NEXT:     %{{.*}} = memref.alloc() : memref<64x64xindex, strided<[2, 4], offset: 6>, 2 : i32>
// CHECK-NEXT:     %{{.*}} = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x64xindex, strided<[2, 4], offset: 6>, 2 : i32>
// CHECK-NEXT:     %base_buffer, %offset, %sizes, %sizes_1, %strides, %strides_1 = memref.extract_strided_metadata %{{.*}} : memref<64x64xindex, strided<[2, 4], offset: 6>, 2 : i32> -> memref<index>, index, index, index, index, index
// CHECK-NEXT:     %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (index, index, index)
// CHECK-NEXT:     %{{.*}} = memref.alloc(%{{.*}}) {alignment = 0 : i64} : memref<?xindex>
// CHECK-NEXT:     %{{.*}} = memref.alloc(%{{.*}}, %{{.*}}, %{{.*}}) {alignment = 0 : i64} : memref<?x?x?xindex>
// CHECK-NEXT:     %{{.*}} = "memref.alloca"(%{{.*}}) <{alignment = 0 : i64, operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xindex>
// CHECK-NEXT:     %{{.*}} = "memref.alloca"(%{{.*}}, %{{.*}}, %{{.*}}) <{alignment = 0 : i64, operandSegmentSizes = array<i32: 3, 0>}> : (index, index, index) -> memref<?x?x?xindex>
// CHECK-NEXT:    %{{.*}} = memref.collapse_shape %{{\S*}}
// CHECK-SAME{LITERAL}: [[0 : i64, 1 : i64]] : memref<10x2xindex> into memref<20xindex>
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : index
// CHECK-NEXT:    %{{.*}} = memref.expand_shape %{{\S*}}
// CHECK-SAME{LITERAL}: [[0 : i64, 1 : i64]] output_shape [%20, 10] : memref<20xindex> into memref<?x10xindex>
// CHECK-NEXT:     %{{.*}} = memref.reinterpret_cast %5 to offset: [0], sizes: [5, 4], strides: [1, 1] : memref<10x2xindex> to memref<5x4xindex>
// CHECK-NEXT:     memref.dealloc %{{.*}} : memref<1xindex>
// CHECK-NEXT:     memref.dealloc %{{.*}} : memref<10x2xindex>
// CHECK-NEXT:     memref.dealloc %{{.*}} : memref<1xindex>
// CHECK-NEXT:     memref.dealloc %{{.*}} : memref<64x64xindex, strided<[2, 4], offset: 6>, 2 : i32>
// CHECK-NEXT:     memref.dealloc %{{.*}} : memref<64x64xindex, strided<[2, 4], offset: 6>, 2 : i32>
// CHECK-NEXT:      %{{.*}} = "test.op"() : () -> memref<32x32xf32>
// CHECK-NEXT:      %{{.*}} = "test.op"() : () -> f32
// CHECK-NEXT:      %{{.*}} = "memref.atomic_rmw"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{kind = 0 : i64}> : (f32, memref<32x32xf32>, index, index) -> f32
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
