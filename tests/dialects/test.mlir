func.func @foo() {
  %0 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<100xf32>
  %1 = arith.constant 0.0 : f32
  "linalg.generic"(%1, %0) ({
  ^0(%2 : f32, %3 : f32):
    "linalg.yield"(%2) : (f32) -> ()
  }) {"indexing_maps" = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], "iterator_types" = [#linalg.iterator_type<parallel>], "operand_segment_sizes" = array<i32: 1, 1>} : (f32, memref<100xf32>) -> ()
  return
}
