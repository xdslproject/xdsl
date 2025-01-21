"builtin.module"() ({
  %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>>
  %1 = "memref.subview"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 2, 0>, static_sizes = array<i64: 4, 4, 4>, static_strides = array<i64: 1, 1, 1>}> : (memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>>) -> memref<4x4x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2 + 8)>>
}) : () -> ()
