"builtin.module"() ({
  "func.func"() <{function_type = (memref<72x72x72xf64>) -> (), sym_name = "test"}> ({
  ^bb0(%arg0: memref<72x72x72xf64>):
    %0 = "memref.subview"(%arg0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 4, 4, 4>, static_sizes = array<i64: 34, 66, 64>, static_strides = array<i64: 1, 1, 1>}> : (memref<72x72x72xf64>) -> memref<34x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
    %1 = "memref.subview"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 31, 0, 0>, static_sizes = array<i64: 1, 64, 64>, static_strides = array<i64: 1, 1, 1>}> : (memref<34x66x64xf64, strided<[5184, 72, 1], offset: 21028>>) -> memref<64x64xf64, strided<[72, 1], offset: 181732>>
    "test.op"(%0, %1) : (memref<34x66x64xf64, strided<[5184, 72, 1], offset: 21028>>, memref<64x64xf64, strided<[72, 1], offset: 181732>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
