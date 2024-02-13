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
// CHECK-NEXT:    "snitch_stream.streaming_region"(%2, %3, %4) <{"stride_patterns" = [#snitch_stream.stride_pattern<ub = [3, 2], strides = [8, 0]>, #snitch_stream.stride_pattern<ub = [3, 2], strides = [0, 8]>, #snitch_stream.stride_pattern<ub = [3, 2], strides = [16, 8]>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
// CHECK-NEXT:  ^0(%a : !stream.readable<!riscv.freg<>>, %b : !stream.readable<!riscv.freg<>>, %c : !stream.writable<!riscv.freg<>>):
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %a : !stream.readable<!riscv.freg<>> to !stream.readable<f64>
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %b : !stream.readable<!riscv.freg<>> to !stream.readable<f64>
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %c : !stream.writable<!riscv.freg<>> to !stream.writable<f64>
// CHECK-NEXT:      "test.op"(%{{.*}}, %{{.*}}, %{{.*}}) : (!stream.readable<f64>, !stream.readable<f64>, !stream.writable<f64>) -> ()
// CHECK-NEXT:  }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()

// CHECK-NEXT:  }
