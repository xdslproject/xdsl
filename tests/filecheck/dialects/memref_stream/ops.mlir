// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK:               builtin.module {
// CHECK-GENERIC:       "builtin.module"() ({

%readable, %writable = "test.op"() : () -> (!stream.readable<f32>, !stream.writable<f32>)

%val = memref_stream.read from %readable : f32
memref_stream.write %val to %writable : f32

// CHECK-NEXT:             %readable, %writable = "test.op"() : () -> (!stream.readable<f32>, !stream.writable<f32>)
// CHECK-NEXT:             %val = memref_stream.read from %readable : f32
// CHECK-NEXT:             memref_stream.write %val to %writable : f32

// CHECK-GENERIC-NEXT:    %readable, %writable = "test.op"() : () -> (!stream.readable<f32>, !stream.writable<f32>)
// CHECK-GENERIC-NEXT:    %val = "memref_stream.read"(%readable) : (!stream.readable<f32>) -> f32
// CHECK-GENERIC-NEXT:    "memref_stream.write"(%val, %writable) : (f32, !stream.writable<f32>) -> ()

%A, %B, %C, %D = "test.op"() : () -> (memref<2xf32>, memref<3xf32>, memref<3x2xf64>, f64)

memref_stream.streaming_region {
    patterns = [
        #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0)>,
        #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d1)>,
        #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0, d1)>
    ]
} ins(%A, %B : memref<2xf32>, memref<3xf32>) outs(%C : memref<3x2xf64>) attrs = {hello = "world"} {
^bb0(%a: !stream.readable<f32>, %b: !stream.readable<f32>, %c: !stream.writable<f64>):
    "test.op"(%a, %b, %c) : (!stream.readable<f32>, !stream.readable<f32>, !stream.writable<f64>) -> ()
}

// CHECK-NEXT:            %A, %B, %C, %D = "test.op"() : () -> (memref<2xf32>, memref<3xf32>, memref<3x2xf64>, f64)
// CHECK-NEXT:            memref_stream.streaming_region {patterns = [#memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0)>, #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d1)>, #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0, d1)>]} ins(%A, %B : memref<2xf32>, memref<3xf32>) outs(%C : memref<3x2xf64>) attrs =  {"hello" = "world"} {
// CHECK-NEXT:            ^0(%a : !stream.readable<f32>, %b : !stream.readable<f32>, %c : !stream.writable<f64>):
// CHECK-NEXT:              "test.op"(%a, %b, %c) : (!stream.readable<f32>, !stream.readable<f32>, !stream.writable<f64>) -> ()
// CHECK-NEXT:            }

// CHECK-GENERIC-NEXT:    %A, %B, %C, %D = "test.op"() : () -> (memref<2xf32>, memref<3xf32>, memref<3x2xf64>, f64)
// CHECK-GENERIC-NEXT:    "memref_stream.streaming_region"(%A, %B, %C) <{"patterns" = [#memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0)>, #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d1)>, #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0, d1)>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^0(%a : !stream.readable<f32>, %b : !stream.readable<f32>, %c : !stream.writable<f64>):
// CHECK-GENERIC-NEXT:      "test.op"(%a, %b, %c) : (!stream.readable<f32>, !stream.readable<f32>, !stream.writable<f64>) -> ()
// CHECK-GENERIC-NEXT:    }) {"hello" = "world"} : (memref<2xf32>, memref<3xf32>, memref<3x2xf64>) -> ()


memref_stream.generic {
    bounds = [#builtin.int<3>, #builtin.int<2>],
    indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
} ins(%A, %B : memref<2xf32>, memref<3xf32>) outs(%C : memref<3x2xf64>) attrs = {hello = "world"} {
^bb0(%arg3: f32, %arg4: f32):
    memref_stream.yield %arg3 : f32
}

// CHECK-NEXT:          memref_stream.generic {bounds = [#builtin.int<3>, #builtin.int<2>], indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%A, %B : memref<2xf32>, memref<3xf32>) outs(%C : memref<3x2xf64>) attrs =  {"hello" = "world"} {
// CHECK-NEXT:          ^1(%arg3 : f32, %arg4 : f32):
// CHECK-NEXT:            memref_stream.yield %arg3 : f32
// CHECK-NEXT:          }

// CHECK-GENERIC-NEXT:    "memref_stream.generic"(%A, %B, %C) <{"bounds" = [#builtin.int<3>, #builtin.int<2>], "indexing_maps" = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], "iterator_types" = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^1(%arg3 : f32, %arg4 : f32):
// CHECK-GENERIC-NEXT:      "memref_stream.yield"(%arg3) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) {"hello" = "world"} : (memref<2xf32>, memref<3xf32>, memref<3x2xf64>) -> ()

memref_stream.generic {
    bounds = [#builtin.int<3>, #builtin.int<2>],
    indexing_maps = [
        affine_map<(d0, d1) -> ()>,
        affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
} ins(%D : f64) outs(%C : memref<3x2xf64>) {
^bb0(%d : f64, %c : f64):
    memref_stream.yield %d : f64
}

// CHECK-NEXT:    memref_stream.generic {bounds = [#builtin.int<3>, #builtin.int<2>], indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%D : f64) outs(%C : memref<3x2xf64>) {
// CHECK-NEXT:    ^2(%d : f64, %c_1 : f64):
// CHECK-NEXT:      memref_stream.yield %d : f64
// CHECK-NEXT:    }

// CHECK-GENERIC-NEXT:    "memref_stream.generic"(%D, %C) <{"bounds" = [#builtin.int<3>, #builtin.int<2>], "indexing_maps" = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], "iterator_types" = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^2(%d : f64, %c_1 : f64):
// CHECK-GENERIC-NEXT:      "memref_stream.yield"(%d) : (f64) -> ()
// CHECK-GENERIC-NEXT:    }) : (f64, memref<3x2xf64>) -> ()

// CHECK-NEXT:          }
// CHECK-GENERIC-NEXT:  }) : () -> ()
