"builtin.module"() ({
  "memref.global"() <{alignment = 64 : i64, initial_value = dense<0> : tensor<1xindex>, sym_name = "g_with_alignment", sym_visibility = "public", type = memref<1xindex>}> : () -> ()
  "memref.global"() <{initial_value = dense<0> : tensor<1xindex>, sym_name = "g", sym_visibility = "public", type = memref<1xindex>}> : () -> ()
  "memref.global"() <{constant, initial_value = dense<0> : tensor<1xindex>, sym_name = "g_constant", sym_visibility = "public", type = memref<1xindex>}> : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "memref_test", sym_visibility = "private"}> ({
    %0 = "memref.get_global"() <{name = @g}> : () -> memref<1xindex>
    %1 = "arith.constant"() <{value = 0 : index}> : () -> index
    %2 = "memref.alloca"() <{alignment = 0 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xindex>
    %3 = "arith.constant"() <{value = 42 : index}> : () -> index
    "memref.store"(%3, %2, %1) : (index, memref<1xindex>, index) -> ()
    %4 = "memref.load"(%2, %1) : (memref<1xindex>, index) -> index
    %5 = "memref.alloc"() <{alignment = 0 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10x2xindex>
    "memref.store"(%3, %5, %3, %4) : (index, memref<10x2xindex>, index, index) -> ()
    %6 = "memref.subview"(%5) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1>, static_strides = array<i64: 1, 1>}> : (memref<10x2xindex>) -> memref<1x1xindex, strided<[2, 1]>>
    %7 = "memref.cast"(%5) : (memref<10x2xindex>) -> memref<?x?xindex>
    %8 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> {i64} : () -> memref<1xindex>
    "memref.copy"(%8, %2) : (memref<1xindex>, memref<1xindex>) -> ()
    "memref.dealloc"(%8) : (memref<1xindex>) -> ()
    "memref.dealloc"(%2) : (memref<1xindex>) -> ()
    "memref.dealloc"(%5) : (memref<10x2xindex>) -> ()
    %9 = "memref.alloc"() <{alignment = 0 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<100xi32, 10>
    %10 = "memref.alloc"() <{alignment = 0 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<100xi32, 9>
    %11 = "memref.alloc"() <{alignment = 0 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<100xi32>
    "memref.dma_start"(%9, %1, %10, %1, %3, %11, %1) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1>} : (memref<100xi32, 10>, index, memref<100xi32, 9>, index, index, memref<100xi32>, index) -> ()
    "memref.dma_wait"(%11, %1, %3) {operandSegmentSizes = array<i32: 1, 1, 1>} : (memref<100xi32>, index, index) -> ()
    %12 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x32xf32>
    %13 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
    %14 = "memref.atomic_rmw"(%13, %12, %1, %1) <{kind = 0 : i64}> : (f32, memref<32x32xf32>, index, index) -> f32
    %15 = "arith.constant"() <{value = 2 : index}> : () -> index
    %16 = "memref.subview"(%5, %15) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 1, 2>, static_strides = array<i64: 1, 1>}> : (memref<10x2xindex>, index) -> memref<2xindex, strided<[1], offset: ?>>
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
