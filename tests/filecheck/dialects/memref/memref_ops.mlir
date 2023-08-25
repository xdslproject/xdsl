// RUN: XDSL_ROUNDTRIP

builtin.module {
  "memref.global"() {"sym_name" = "g", "type" = memref<1xindex>, "initial_value" = dense<0> : tensor<1xindex>, "sym_visibility" = "public"} : () -> ()
  func.func private @memref_test() {
    %0 = "memref.get_global"() {"name" = @g} : () -> memref<1xindex>
    %1 = arith.constant 0 : index
    %2 = "memref.alloca"() {"alignment" = 0 : i64, "operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<1xindex>
    %3 = arith.constant 42 : index
    "memref.store"(%3, %2, %1) : (index, memref<1xindex>, index) -> ()
    %4 = "memref.load"(%2, %1) : (memref<1xindex>, index) -> index
    %5 = "memref.alloc"() {"alignment" = 0 : i64, "operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<10x2xindex>
    "memref.store"(%3, %5, %3, %4) : (index, memref<10x2xindex>, index, index) -> ()
    %6 = "memref.subview"(%5) {"operand_segment_sizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 0, 0>, "static_sizes" = array<i64: 1, 1>, "static_strides" = array<i64: 1, 1>} : (memref<10x2xindex>) -> memref<1x1xindex>
    %7 = "memref.cast"(%5) : (memref<10x2xindex>) -> memref<?x?xindex>
    %8 = "memref.alloca"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<1xindex>
    "memref.dealloc"(%2) : (memref<1xindex>) -> ()
    "memref.dealloc"(%5) : (memref<10x2xindex>) -> ()
    "memref.dealloc"(%8) : (memref<1xindex>) -> ()
    func.return
  }
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   "memref.global"() {"sym_name" = "g", "type" = memref<1xindex>, "initial_value" = dense<0> : tensor<1xindex>, "sym_visibility" = "public"} : () -> ()
// CHECK-NEXT:   func.func private @memref_test() {
// CHECK-NEXT:     %{{.*}} = "memref.get_global"() {"name" = @g} : () -> memref<1xindex>
// CHECK-NEXT:     %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:     %{{.*}} = "memref.alloca"() {"alignment" = 0 : i64, "operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<1xindex>
// CHECK-NEXT:     %{{.*}} = arith.constant 42 : index
// CHECK-NEXT:     "memref.store"(%{{.*}}, %{{.*}}, %{{.*}}) : (index, memref<1xindex>, index) -> ()
// CHECK-NEXT:     %{{.*}} = "memref.load"(%{{.*}}, %{{.*}}) : (memref<1xindex>, index) -> index
// CHECK-NEXT:     %{{.*}} = "memref.alloc"() {"alignment" = 0 : i64, "operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<10x2xindex>
// CHECK-NEXT:     "memref.store"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (index, memref<10x2xindex>, index, index) -> ()
// CHECK-NEXT:     %{{.*}} = "memref.subview"(%5) {"operand_segment_sizes" = array<i32: 1, 0, 0, 0>, "static_offsets" = array<i64: 0, 0>, "static_sizes" = array<i64: 1, 1>, "static_strides" = array<i64: 1, 1>} : (memref<10x2xindex>) -> memref<1x1xindex>
// CHECK-NEXT:     %{{.*}} = "memref.cast"(%{{.*}}) : (memref<10x2xindex>) -> memref<?x?xindex>
// CHECK-NEXT:     %{{.*}} = "memref.alloca"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<1xindex>
// CHECK-NEXT:     "memref.dealloc"(%{{.*}}) : (memref<1xindex>) -> ()
// CHECK-NEXT:     "memref.dealloc"(%{{.*}}) : (memref<10x2xindex>) -> ()
// CHECK-NEXT:     "memref.dealloc"(%{{.*}}) : (memref<1xindex>) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
