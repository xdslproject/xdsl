// RUN: xdsl-opt %s -p convert-linalg-to-stream | filecheck %s

%X, %Y, %Z = "test.op"() : () -> (memref<8x16xf64>, memref<8x16xf64>, memref<8x16xf64>)

linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%X, %Y : memref<8x16xf64>, memref<8x16xf64>) outs(%Z : memref<8x16xf64>) {
^0(%x : f64, %y : f64, %z : f64):
    %r0 = arith.addf %x, %y : f64
    linalg.yield %r0 : f64
}


// CHECK:       builtin.module {
// CHECK-NEXT:    %X, %Y, %Z = "test.op"() : () -> (memref<8x16xf64>, memref<8x16xf64>, memref<8x16xf64>)
// CHECK-NEXT:    %0 = "stream.stride_pattern"() {"ub" = [#builtin.int<8>, #builtin.int<16>], "strides" = [#builtin.int<16>, #builtin.int<1>], "dm" = #builtin.int<31>} : () -> !stream.stride_pattern_type<2>
// CHECK-NEXT:    %1 = arith.constant 128 : index
// CHECK-NEXT:    "stream.generic"(%1, %X, %Y, %Z, %0) <{"operandSegmentSizes" = array<i32: 1, 2, 1, 1>}> ({
// CHECK-NEXT:    ^0(%x : f64, %y : f64):
// CHECK-NEXT:      %r0 = arith.addf %x, %y : f64
// CHECK-NEXT:      "stream.yield"(%r0) : (f64) -> ()
// CHECK-NEXT:    }) : (index, memref<8x16xf64>, memref<8x16xf64>, memref<8x16xf64>, !stream.stride_pattern_type<2>) -> ()
// CHECK-NEXT:  }
