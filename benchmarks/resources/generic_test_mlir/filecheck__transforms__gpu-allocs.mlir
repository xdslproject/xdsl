"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "memref_test", sym_visibility = "private"}> ({
    %0 = "test.op"() : () -> index
    %1 = "test.op"() : () -> index
    %2 = "test.op"() : () -> index
    %3 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x64xindex, strided<[2, 4], offset: 6>, 2 : i32>
    %4 = "memref.alloc"(%0) <{alignment = 0 : i64, operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xindex>
    %5 = "memref.alloc"(%0, %1, %2) <{alignment = 0 : i64, operandSegmentSizes = array<i32: 3, 0>}> : (index, index, index) -> memref<?x?x?xindex>
    "memref.dealloc"(%3) : (memref<64x64xindex, strided<[2, 4], offset: 6>, 2 : i32>) -> ()
    "memref.dealloc"(%4) : (memref<?xindex>) -> ()
    "memref.dealloc"(%5) : (memref<?x?x?xindex>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
