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

%A, %B, %C = "test.op"() : () -> (memref<2xf32>, memref<3xf32>, memref<3x2xf64>)

memref_stream.streaming_region {
    bounds = [3, 2],
    indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
    ]
} ins(%A, %B : memref<2xf32>, memref<3xf32>) outs(%C : memref<3x2xf64>) attrs = {hello = "world"} {
^bb0(%a: !stream.readable<f32>, %b: !stream.readable<f32>, %c: !stream.writable<f64>):
    "test.op"(%a, %b, %c) : (!stream.readable<f32>, !stream.readable<f32>, !stream.writable<f64>) -> ()
}

// CHECK-NEXT:            %A, %B, %C = "test.op"() : () -> (memref<2xf32>, memref<3xf32>, memref<3x2xf64>)
// CHECK-NEXT:            memref_stream.streaming_region {bounds = [3, 2], indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>]} ins(%A, %B : memref<2xf32>, memref<3xf32>) outs(%C : memref<3x2xf64>) attrs =  {"hello" = "world"} {
// CHECK-NEXT:            ^0(%a : !stream.readable<f32>, %b : !stream.readable<f32>, %c : !stream.writable<f64>):
// CHECK-NEXT:              "test.op"(%a, %b, %c) : (!stream.readable<f32>, !stream.readable<f32>, !stream.writable<f64>) -> ()
// CHECK-NEXT:            }

// CHECK-GENERIC-NEXT:    %A, %B, %C = "test.op"() : () -> (memref<2xf32>, memref<3xf32>, memref<3x2xf64>)
// CHECK-GENERIC-NEXT:    "memref_stream.streaming_region"(%A, %B, %C) <{"bounds" = [#builtin.int<3>, #builtin.int<2>], "indexing_maps" = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^0(%a : !stream.readable<f32>, %b : !stream.readable<f32>, %c : !stream.writable<f64>):
// CHECK-GENERIC-NEXT:      "test.op"(%a, %b, %c) : (!stream.readable<f32>, !stream.readable<f32>, !stream.writable<f64>) -> ()
// CHECK-GENERIC-NEXT:    }) {"hello" = "world"} : (memref<2xf32>, memref<3xf32>, memref<3x2xf64>) -> ()

// CHECK-NEXT:          }
// CHECK-GENERIC-NEXT:  }) : () -> ()
