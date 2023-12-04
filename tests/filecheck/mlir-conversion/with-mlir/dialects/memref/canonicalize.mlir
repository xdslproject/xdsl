//RUN: xdsl-opt %s -p canonicalize | mlir-opt | filecheck %s

builtin.module {
    func.func @test(%ref: memref<72x72x72xf64>) {
        // source:
        %5 = "memref.subview"(%ref) <{
            "static_offsets" = array<i64: 4, 4, 4>,
            "static_sizes" = array<i64: 34, 66, 64>,
            "static_strides" = array<i64: 1, 1, 1>,
            "operandSegmentSizes" = array<i32: 1, 0, 0, 0>
        }> : (memref<72x72x72xf64>) -> memref<34x66x64xf64, strided<[5184, 72, 1], offset: 21028>>
        // non-canonicalized
        %33 = "memref.subview"(%5) <{
            "static_offsets" = array<i64: 31, 0, 0>,
            "static_sizes" = array<i64: 1, 64, 64>,
            "static_strides" = array<i64: 1, 1, 1>,
            "operandSegmentSizes" = array<i32: 1, 0, 0, 0>
        }> : (memref<34x66x64xf64, strided<[5184, 72, 1], offset: 21028>>) -> memref<64x64xf64, strided<[72, 1], offset: 181732>>

        func.return
    }
}
