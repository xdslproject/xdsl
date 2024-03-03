// RUN: xdsl-opt -p convert-memref-stream-to-snitch %s | filecheck %s

// CHECK:       builtin.module {

%readable, %writable = "test.op"() : () -> (!stream.readable<f32>, !stream.writable<f32>)

%val = memref_stream.read from %readable : f32
memref_stream.write %val to %writable : f32

// CHECK-NEXT:    %readable, %writable = "test.op"() : () -> (!stream.readable<f32>, !stream.writable<f32>)
// CHECK-NEXT:    %val = builtin.unrealized_conversion_cast %readable : !stream.readable<f32> to !stream.readable<!riscv.freg<>>
// CHECK-NEXT:    %{{.*}} = riscv_snitch.read from %val : !riscv.freg<>
// CHECK-NEXT:    %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.freg<> to f32
// CHECK-NEXT:    %{{.*}} = builtin.unrealized_conversion_cast %writable : !stream.writable<f32> to !stream.writable<!riscv.freg<>>
// CHECK-NEXT:    %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : f32 to !riscv.freg<>
// CHECK-NEXT:    riscv_snitch.write %{{.*}} to %{{.*}} : !riscv.freg<>

%A, %B, %C = "test.op"() : () -> (memref<2xf64>, memref<3xf64>, memref<3x2xf64>)

memref_stream.streaming_region {
    bounds = [3, 2],
    indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
    ]
} ins(%A, %B : memref<2xf64>, memref<3xf64>) outs(%C : memref<3x2xf64>) attrs = {hello = "world"} {
^bb0(%a: !stream.readable<f64>, %b: !stream.readable<f64>, %c: !stream.writable<f64>):
    "test.op"(%a, %b, %c) : (!stream.readable<f64>, !stream.readable<f64>, !stream.writable<f64>) -> ()
}

// CHECK-NEXT:  %A, %B, %C = "test.op"() : () -> (memref<2xf64>, memref<3xf64>, memref<3x2xf64>)
// CHECK-NEXT:  %{{.*}} = builtin.unrealized_conversion_cast %A : memref<2xf64> to !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = builtin.unrealized_conversion_cast %B : memref<3xf64> to !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = builtin.unrealized_conversion_cast %C : memref<3x2xf64> to !riscv.reg<>
// CHECK-NEXT:    "snitch_stream.streaming_region"(%2, %3, %4) <{"stride_patterns" = [#snitch_stream.stride_pattern<ub = [2, 3], strides = [0, 8]>, #snitch_stream.stride_pattern<ub = [2, 3], strides = [8, 0]>, #snitch_stream.stride_pattern<ub = [6], strides = [8]>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
// CHECK-NEXT:  ^0(%a : !stream.readable<!riscv.freg<>>, %b : !stream.readable<!riscv.freg<>>, %c : !stream.writable<!riscv.freg<>>):
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %a : !stream.readable<!riscv.freg<>> to !stream.readable<f64>
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %b : !stream.readable<!riscv.freg<>> to !stream.readable<f64>
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %c : !stream.writable<!riscv.freg<>> to !stream.writable<f64>
// CHECK-NEXT:      "test.op"(%{{.*}}, %{{.*}}, %{{.*}}) : (!stream.readable<f64>, !stream.readable<f64>, !stream.writable<f64>) -> ()
// CHECK-NEXT:  }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()

memref_stream.streaming_region {
    bounds = [3, 2],
    indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
    ]
} ins(%C, %C : memref<3x2xf64>, memref<3x2xf64>) {
^bb0(%c0: !stream.readable<f64>, %c1: !stream.readable<f64>):
    "test.op"(%c0, %c1) : (!stream.readable<f64>, !stream.readable<f64>) -> ()
}

// CHECK-NEXT:  %{{.*}} = builtin.unrealized_conversion_cast %C : memref<3x2xf64> to !riscv.reg<>
// CHECK-NEXT:  %{{.*}} = builtin.unrealized_conversion_cast %C : memref<3x2xf64> to !riscv.reg<>
// CHECK-NEXT:    "snitch_stream.streaming_region"(%{{.*}}, %{{.*}}) <{"stride_patterns" = [#snitch_stream.stride_pattern<ub = [6], strides = [8]>], "operandSegmentSizes" = array<i32: 2, 0>}> ({
// CHECK-NEXT:  ^{{.*}}(%c0 : !stream.readable<!riscv.freg<>>, %c1 : !stream.readable<!riscv.freg<>>):
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %c0 : !stream.readable<!riscv.freg<>> to !stream.readable<f64>
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %c1 : !stream.readable<!riscv.freg<>> to !stream.readable<f64>
// CHECK-NEXT:      "test.op"(%{{.*}}, %{{.*}}) : (!stream.readable<f64>, !stream.readable<f64>) -> ()
// CHECK-NEXT:  }) : (!riscv.reg<>, !riscv.reg<>) -> ()


%D, %E = "test.op"() : () -> (memref<1x1x8x8xf64>, memref<1x1x3x3xf64>)
// CHECK-NEXT:   %D, %E = "test.op"() : () -> (memref<1x1x8x8xf64>, memref<1x1x3x3xf64>)

memref_stream.streaming_region {
    bounds = [1, 1, 6, 6, 1, 3, 3],
    indexing_maps = [
    affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>,
    affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
    ]
} ins(%D, %E : memref<1x1x8x8xf64>, memref<1x1x3x3xf64>) {
^0(%d_stream : !stream.readable<f64>, %e_stream : !stream.readable<f64>):
"test.op"(%d_stream, %e_stream) : (!stream.readable<f64>, !stream.readable<f64>) -> ()
}

// CHECK-NEXT:    %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : memref<1x1x8x8xf64> to !riscv.reg<>
// CHECK-NEXT:    %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : memref<1x1x3x3xf64> to !riscv.reg<>
// CHECK-NEXT:    "snitch_stream.streaming_region"(%{{.*}}, %{{.*}}) <{"stride_patterns" = [#snitch_stream.stride_pattern<ub = [3, 3, 6, 6], strides = [8, 64, 8, 64]>, #snitch_stream.stride_pattern<ub = [3, 3, 36], strides = [8, 24, 0]>], "operandSegmentSizes" = array<i32: 2, 0>}> ({
// CHECK-NEXT:    ^{{.*}}(%{{.*}} : !stream.readable<!riscv.freg<>>, %{{.*}} : !stream.readable<!riscv.freg<>>):
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !stream.readable<!riscv.freg<>> to !stream.readable<f64>
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !stream.readable<!riscv.freg<>> to !stream.readable<f64>
// CHECK-NEXT:      "test.op"(%{{.*}}, %{{.*}}) : (!stream.readable<f64>, !stream.readable<f64>) -> ()
// CHECK-NEXT:    }) : (!riscv.reg<>, !riscv.reg<>) -> ()

// CHECK-NEXT:  }
