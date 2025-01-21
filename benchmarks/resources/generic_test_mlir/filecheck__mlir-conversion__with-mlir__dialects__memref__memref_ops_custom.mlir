"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "memref_alloca_scope"}> ({
    "memref.alloca_scope"() ({
      "memref.alloca_scope.return"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  %0 = "test.op"() : () -> i32
  %1 = "test.op"() : () -> index
  %2 = "test.op"() : () -> index
  %3 = "test.op"() : () -> memref<2x3xi32>
  %4 = "test.op"() : () -> memref<10x3xi32>
  "memref.store"(%0, %3, %1, %2) : (i32, memref<2x3xi32>, index, index) -> ()
  "memref.store"(%0, %3, %1, %2) <{nontemporal = false}> : (i32, memref<2x3xi32>, index, index) -> ()
  "memref.store"(%0, %3, %1, %2) <{nontemporal = true}> : (i32, memref<2x3xi32>, index, index) -> ()
  %5 = "memref.load"(%3, %1, %2) : (memref<2x3xi32>, index, index) -> i32
  %6 = "memref.load"(%3, %1, %2) <{nontemporal = false}> : (memref<2x3xi32>, index, index) -> i32
  %7 = "memref.load"(%3, %1, %2) <{nontemporal = true}> : (memref<2x3xi32>, index, index) -> i32
  %8 = "memref.expand_shape"(%4) <{reassociation = [[0, 1], [2]], static_output_shape = array<i64: 5, 2, 3>}> : (memref<10x3xi32>) -> memref<5x2x3xi32>
  %9 = "memref.collapse_shape"(%4) <{reassociation = [[0, 1]]}> : (memref<10x3xi32>) -> memref<30xi32>
  %10 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x3xf32>
  %11 = "memref.alloc"(%2) <{alignment = 8 : i64, operandSegmentSizes = array<i32: 0, 1>}> : (index) -> memref<2x3xf32, affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>, 1>
  "memref.dealloc"(%10) : (memref<2x3xf32>) -> ()
  %12 = "memref.subview"(%4) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 3>, static_strides = array<i64: 1, 1>}> : (memref<10x3xi32>) -> memref<3xi32>
  %13 = "memref.subview"(%4, %1) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, static_offsets = array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 1, 3>, static_strides = array<i64: 1, 1>}> : (memref<10x3xi32>, index) -> memref<3xi32, strided<[1], offset: ?>>
}) : () -> ()
