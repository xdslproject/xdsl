"builtin.module"() ({
  %0 = "test.op"() : () -> f64
  %1:3 = "test.op"() : () -> (index, index, index)
  %2 = "test.op"() : () -> memref<4x3x2xf64>
  %3 = "memref.subview"(%2) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 3, 2>, static_strides = array<i64: 1, 1, 1>}> : (memref<4x3x2xf64>) -> memref<3x2xf64>
  %4 = "memref.subview"(%2) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 3, 2>, static_strides = array<i64: 1, 1, 1>}> : (memref<4x3x2xf64>) -> memref<3x2xf64, strided<[2, 1], offset: 6>>
  %5 = "memref.subview"(%2, %1#2) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 3, 2>, static_strides = array<i64: 1, 1, 1>}> : (memref<4x3x2xf64>, index) -> memref<3x2xf64, strided<[2, 1], offset: ?>>
  %6 = "test.op"() : () -> memref<5x4x3x2xf64>
  %7 = "memref.subview"(%6, %1#2, %1#2) <{operandSegmentSizes = array<i32: 1, 2, 0, 0>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808, 0, 0>, static_sizes = array<i64: 1, 1, 3, 2>, static_strides = array<i64: 1, 1, 1, 1>}> : (memref<5x4x3x2xf64>, index, index) -> memref<3x2xf64, strided<[2, 1], offset: ?>>
}) : () -> ()
