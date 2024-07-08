// RUN: xdsl-opt -p convert-memref-stream-to-snitch-stream %s | filecheck %s

// CHECK:       builtin.module {

%readable, %writable = "test.op"() : () -> (!stream.readable<f32>, !stream.writable<f32>)

%val = memref_stream.read from %readable : f32
memref_stream.write %val to %writable : f32

// CHECK-NEXT:    %readable, %writable = "test.op"() : () -> (!stream.readable<f32>, !stream.writable<f32>)
// CHECK-NEXT:    %val = builtin.unrealized_conversion_cast %readable : !stream.readable<f32> to !stream.readable<!riscv.freg>
// CHECK-NEXT:    %{{.*}} = riscv_snitch.read from %val : !riscv.freg
// CHECK-NEXT:    %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.freg to f32
// CHECK-NEXT:    %{{.*}} = builtin.unrealized_conversion_cast %writable : !stream.writable<f32> to !stream.writable<!riscv.freg>
// CHECK-NEXT:    %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : f32 to !riscv.freg
// CHECK-NEXT:    %{{.*}} = riscv.fmv.s %{{.*}} : (!riscv.freg) -> !riscv.freg
// CHECK-NEXT:    riscv_snitch.write %{{.*}} to %{{.*}} : !riscv.freg

%A, %B, %C = "test.op"() : () -> (memref<2xf64>, memref<3xf64>, memref<3x2xf64>)

memref_stream.streaming_region {
    patterns = [
        #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0)>,
        #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d1)>,
        #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0, d1)>
    ]
} ins(%A, %B : memref<2xf64>, memref<3xf64>) outs(%C : memref<3x2xf64>) attrs = {hello = "world"} {
^bb0(%a: !stream.readable<f64>, %b: !stream.readable<f64>, %c: !stream.writable<f64>):
    "test.op"(%a, %b, %c) : (!stream.readable<f64>, !stream.readable<f64>, !stream.writable<f64>) -> ()
}

// CHECK-NEXT:  %A, %B, %C = "test.op"() : () -> (memref<2xf64>, memref<3xf64>, memref<3x2xf64>)
// CHECK-NEXT:    %A_1 = builtin.unrealized_conversion_cast %A : memref<2xf64> to !riscv.reg
// CHECK-NEXT:    %B_1 = builtin.unrealized_conversion_cast %B : memref<3xf64> to !riscv.reg
// CHECK-NEXT:    %C_1 = builtin.unrealized_conversion_cast %C : memref<3x2xf64> to !riscv.reg
// CHECK-NEXT:    snitch_stream.streaming_region {
// CHECK-NEXT:      patterns = [
// CHECK-NEXT:        #snitch_stream.stride_pattern<ub = [3, 2], strides = [8, 0]>,
// CHECK-NEXT:        #snitch_stream.stride_pattern<ub = [3, 2], strides = [0, 8]>,
// CHECK-NEXT:        #snitch_stream.stride_pattern<ub = [6], strides = [8]>
// CHECK-NEXT:      ]
// CHECK-NEXT:    } ins(%A_1, %B_1 : !riscv.reg, !riscv.reg) outs(%C_1 : !riscv.reg) {
// CHECK-NEXT:    ^{{.*}}(%a : !stream.readable<!riscv.freg>, %b : !stream.readable<!riscv.freg>, %c : !stream.writable<!riscv.freg>):
// CHECK-NEXT:      %a_1 = builtin.unrealized_conversion_cast %a : !stream.readable<!riscv.freg> to !stream.readable<f64>
// CHECK-NEXT:      %b_1 = builtin.unrealized_conversion_cast %b : !stream.readable<!riscv.freg> to !stream.readable<f64>
// CHECK-NEXT:      %c_1 = builtin.unrealized_conversion_cast %c : !stream.writable<!riscv.freg> to !stream.writable<f64>
// CHECK-NEXT:      "test.op"(%a_1, %b_1, %c_1) : (!stream.readable<f64>, !stream.readable<f64>, !stream.writable<f64>) -> ()
// CHECK-NEXT:    }

memref_stream.streaming_region {
    patterns = [
        #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0, d1)>,
        #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0, d1)>
    ]
} ins(%C, %C : memref<3x2xf64>, memref<3x2xf64>) {
^bb0(%c0: !stream.readable<f64>, %c1: !stream.readable<f64>):
    "test.op"(%c0, %c1) : (!stream.readable<f64>, !stream.readable<f64>) -> ()
}

// CHECK-NEXT:    %C_2 = builtin.unrealized_conversion_cast %C : memref<3x2xf64> to !riscv.reg
// CHECK-NEXT:    %C_3 = builtin.unrealized_conversion_cast %C : memref<3x2xf64> to !riscv.reg
// CHECK-NEXT:    snitch_stream.streaming_region {
// CHECK-NEXT:      patterns = [
// CHECK-NEXT:        #snitch_stream.stride_pattern<ub = [6], strides = [8]>
// CHECK-NEXT:      ]
// CHECK-NEXT:    } ins(%C_2, %C_3 : !riscv.reg, !riscv.reg) {
// CHECK-NEXT:    ^{{.*}}(%c0 : !stream.readable<!riscv.freg>, %c1 : !stream.readable<!riscv.freg>):
// CHECK-NEXT:      %c0_1 = builtin.unrealized_conversion_cast %c0 : !stream.readable<!riscv.freg> to !stream.readable<f64>
// CHECK-NEXT:      %c1_1 = builtin.unrealized_conversion_cast %c1 : !stream.readable<!riscv.freg> to !stream.readable<f64>
// CHECK-NEXT:      "test.op"(%c0_1, %c1_1) : (!stream.readable<f64>, !stream.readable<f64>) -> ()
// CHECK-NEXT:    }

%D, %E = "test.op"() : () -> (memref<1x1x8x8xf64>, memref<1x1x3x3xf64>)
// CHECK-NEXT:   %D, %E = "test.op"() : () -> (memref<1x1x8x8xf64>, memref<1x1x3x3xf64>)

memref_stream.streaming_region {
    patterns = [
        #memref_stream.stride_pattern<ub = [1, 1, 6, 6, 1, 3, 3], index_map = (d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>,
        #memref_stream.stride_pattern<ub = [1, 1, 6, 6, 1, 3, 3], index_map = (d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
    ]
} ins(%D, %E : memref<1x1x8x8xf64>, memref<1x1x3x3xf64>) {
^0(%d_stream : !stream.readable<f64>, %e_stream : !stream.readable<f64>):
    "test.op"(%d_stream, %e_stream) : (!stream.readable<f64>, !stream.readable<f64>) -> ()
}

// CHECK-NEXT:    %D_1 = builtin.unrealized_conversion_cast %D : memref<1x1x8x8xf64> to !riscv.reg
// CHECK-NEXT:    %E_1 = builtin.unrealized_conversion_cast %E : memref<1x1x3x3xf64> to !riscv.reg
// CHECK-NEXT:    snitch_stream.streaming_region {
// CHECK-NEXT:      patterns = [
// CHECK-NEXT:        #snitch_stream.stride_pattern<ub = [6, 6, 3, 3], strides = [64, 8, 64, 8]>,
// CHECK-NEXT:        #snitch_stream.stride_pattern<ub = [36, 3, 3], strides = [0, 24, 8]>
// CHECK-NEXT:      ]
// CHECK-NEXT:    } ins(%D_1, %E_1 : !riscv.reg, !riscv.reg) {
// CHECK-NEXT:    ^{{.*}}(%d_stream : !stream.readable<!riscv.freg>, %e_stream : !stream.readable<!riscv.freg>):
// CHECK-NEXT:      %d_stream_1 = builtin.unrealized_conversion_cast %d_stream : !stream.readable<!riscv.freg> to !stream.readable<f64>
// CHECK-NEXT:      %e_stream_1 = builtin.unrealized_conversion_cast %e_stream : !stream.readable<!riscv.freg> to !stream.readable<f64>
// CHECK-NEXT:      "test.op"(%d_stream_1, %e_stream_1) : (!stream.readable<f64>, !stream.readable<f64>) -> ()
// CHECK-NEXT:    }

%F = "test.op"() : () -> memref<8x8xf64>
// CHECK-NEXT:   %F = "test.op"() : () -> memref<8x8xf64>

memref_stream.streaming_region {
    patterns = [
        #memref_stream.stride_pattern<ub = [8, 8, 8], index_map = (m, n, k) -> (m, k)>,
        #memref_stream.stride_pattern<ub = [8, 8, 8], index_map = (m, n, k) -> (k, n)>,
        #memref_stream.stride_pattern<ub = [8, 8], index_map = (m, n) -> (m, n)>
    ]
} ins(%F, %F, %F : memref<8x8xf64>, memref<8x8xf64>, memref<8x8xf64>) {
^0(%x_stream : !stream.readable<f64>, %w_stream : !stream.readable<f64>, %b_stream : !stream.readable<f64>):
    "test.op"(%x_stream, %w_stream, %b_stream) : (!stream.readable<f64>, !stream.readable<f64>, !stream.readable<f64>) -> ()
}

// CHECK-NEXT:    %F_1 = builtin.unrealized_conversion_cast %F : memref<8x8xf64> to !riscv.reg
// CHECK-NEXT:    %F_2 = builtin.unrealized_conversion_cast %F : memref<8x8xf64> to !riscv.reg
// CHECK-NEXT:    %F_3 = builtin.unrealized_conversion_cast %F : memref<8x8xf64> to !riscv.reg
// CHECK-NEXT:    snitch_stream.streaming_region {
// CHECK-NEXT:      patterns = [
// CHECK-NEXT:        #snitch_stream.stride_pattern<ub = [8, 8, 8], strides = [64, 0, 8]>,
// CHECK-NEXT:        #snitch_stream.stride_pattern<ub = [8, 8, 8], strides = [0, 8, 64]>,
// CHECK-NEXT:        #snitch_stream.stride_pattern<ub = [64], strides = [8]>
// CHECK-NEXT:      ]
// CHECK-NEXT:    } ins(%F_1, %F_2, %F_3 : !riscv.reg, !riscv.reg, !riscv.reg) {
// CHECK-NEXT:    ^{{.*}}(%x_stream : !stream.readable<!riscv.freg>, %w_stream : !stream.readable<!riscv.freg>, %b_stream : !stream.readable<!riscv.freg>):
// CHECK-NEXT:      %x_stream_1 = builtin.unrealized_conversion_cast %x_stream : !stream.readable<!riscv.freg> to !stream.readable<f64>
// CHECK-NEXT:      %w_stream_1 = builtin.unrealized_conversion_cast %w_stream : !stream.readable<!riscv.freg> to !stream.readable<f64>
// CHECK-NEXT:      %b_stream_1 = builtin.unrealized_conversion_cast %b_stream : !stream.readable<!riscv.freg> to !stream.readable<f64>
// CHECK-NEXT:      "test.op"(%x_stream_1, %w_stream_1, %b_stream_1) : (!stream.readable<f64>, !stream.readable<f64>, !stream.readable<f64>) -> ()
// CHECK-NEXT:    }

%G, %H = "test.op"() : () -> (f64, memref<16x16xf64>)
// CHECK-NEXT:   %G, %H = "test.op"() : () -> (f64, memref<16x16xf64>)

memref_stream.streaming_region {
    patterns = [
    #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>
    ]
} outs(%H : memref<16x16xf64>) {
^0(%h_stream : !stream.writable<f64>):
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c256 = arith.constant 256 : i32
    scf.for %i = %c0 to %c256 step %c1 {
        memref_stream.write %G to %h_stream : f64
    }
}

// CHECK-NEXT:    %H_1 = builtin.unrealized_conversion_cast %H : memref<16x16xf64> to !riscv.reg
// CHECK-NEXT:    snitch_stream.streaming_region {
// CHECK-NEXT:      patterns = [
// CHECK-NEXT:        #snitch_stream.stride_pattern<ub = [256], strides = [8]>
// CHECK-NEXT:      ]
// CHECK-NEXT:    } outs(%H_1 : !riscv.reg) {
// CHECK-NEXT:    ^{{.*}}(%h_stream : !stream.writable<!riscv.freg>):
// CHECK-NEXT:      %h_stream_1 = builtin.unrealized_conversion_cast %h_stream : !stream.writable<!riscv.freg> to !stream.writable<f64>
// CHECK-NEXT:      %c0_2 = arith.constant 0 : i32
// CHECK-NEXT:      %c1_2 = arith.constant 1 : i32
// CHECK-NEXT:      %c256 = arith.constant 256 : i32
// CHECK-NEXT:      scf.for %i = %c0_2 to %c256 step %c1_2 : i32 {
// CHECK-NEXT:        %{{.*}} = builtin.unrealized_conversion_cast %h_stream_1 : !stream.writable<f64> to !stream.writable<!riscv.freg>
// CHECK-NEXT:        %{{.*}} = builtin.unrealized_conversion_cast %G : f64 to !riscv.freg
// CHECK-NEXT:        %{{.*}} = riscv.fmv.d %{{.*}} : (!riscv.freg) -> !riscv.freg
// CHECK-NEXT:        riscv_snitch.write %{{.*}} to %{{.*}} : !riscv.freg
// CHECK-NEXT:      }
// CHECK-NEXT:    }

%I, %J, %K = "test.op"() : () -> (memref<3x5xf64>, memref<5x8xf64>, memref<3x8xf64>)
// CHECK-NEXT:    %I, %J, %K = "test.op"() : () -> (memref<3x5xf64>, memref<5x8xf64>, memref<3x8xf64>)
// CHECK-NEXT:    %I_1 = builtin.unrealized_conversion_cast %I : memref<3x5xf64> to !riscv.reg
// CHECK-NEXT:    %J_1 = builtin.unrealized_conversion_cast %J : memref<5x8xf64> to !riscv.reg
// CHECK-NEXT:    %K_1 = builtin.unrealized_conversion_cast %K : memref<3x8xf64> to !riscv.reg

// more complex maps
memref_stream.streaming_region {
    patterns = [
        #memref_stream.stride_pattern<ub = [3, 2, 5, 4], index_map = (d0, d1, d2, d3) -> (d0, d2)>,
        #memref_stream.stride_pattern<ub = [3, 2, 5, 4], index_map = (d0, d1, d2, d3) -> (d2, ((d1 * 4) + d3))>,
        #memref_stream.stride_pattern<ub = [3, 2, 4], index_map = (d0, d1, d2) -> (d0, ((d1 * 4) + d2))>
    ]
} ins(%I, %J : memref<3x5xf64>, memref<5x8xf64>) outs(%K : memref<3x8xf64>) {
^0(%i : !stream.readable<f64>, %j : !stream.readable<f64>, %k : !stream.writable<f64>):
    %res = "test.op"() : () -> f64
    memref_stream.yield %res : f64
}
// CHECK-NEXT:    snitch_stream.streaming_region {
// CHECK-NEXT:      patterns = [
// CHECK-NEXT:        #snitch_stream.stride_pattern<ub = [3, 2, 5, 4], strides = [40, 0, 8, 0]>,
// CHECK-NEXT:        #snitch_stream.stride_pattern<ub = [3, 2, 5, 4], strides = [0, 32, 64, 8]>,
// CHECK-NEXT:        #snitch_stream.stride_pattern<ub = [24], strides = [8]>
// CHECK-NEXT:      ]
// CHECK-NEXT:    } ins(%I_1, %J_1 : !riscv.reg, !riscv.reg) outs(%K_1 : !riscv.reg) {
// CHECK-NEXT:    ^{{.*}}(%i_1 : !stream.readable<!riscv.freg>, %j : !stream.readable<!riscv.freg>, %k : !stream.writable<!riscv.freg>):
// CHECK-NEXT:      %i_2 = builtin.unrealized_conversion_cast %i_1 : !stream.readable<!riscv.freg> to !stream.readable<f64>
// CHECK-NEXT:      %j_1 = builtin.unrealized_conversion_cast %j : !stream.readable<!riscv.freg> to !stream.readable<f64>
// CHECK-NEXT:      %k_1 = builtin.unrealized_conversion_cast %k : !stream.writable<!riscv.freg> to !stream.writable<f64>
// CHECK-NEXT:      %res = "test.op"() : () -> f64
// CHECK-NEXT:      memref_stream.yield %res : f64
// CHECK-NEXT:    }


%A_strided = "test.op"() : () -> memref<3x2xf64, strided<[4, 1]>>
// CHECK-NEXT:    %A_strided = "test.op"() : () -> memref<3x2xf64, strided<[4, 1]>>


memref_stream.streaming_region {
    patterns = [
        #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0, d1)>
    ]
} ins(%A_strided : memref<3x2xf64, strided<[4, 1]>>) {
^bb0(%a_strided: !stream.readable<f64>):
    "test.op"(%a_strided) : (!stream.readable<f64>) -> ()
}

// CHECK-NEXT:    %A_strided_1 = builtin.unrealized_conversion_cast %A_strided : memref<3x2xf64, strided<[4, 1]>> to !riscv.reg
// CHECK-NEXT:    snitch_stream.streaming_region {
// CHECK-NEXT:      patterns = [
// CHECK-NEXT:        #snitch_stream.stride_pattern<ub = [3, 2], strides = [32, 8]>
// CHECK-NEXT:      ]
// CHECK-NEXT:    } ins(%A_strided_1 : !riscv.reg) {
// CHECK-NEXT:    ^{{.*}}(%a_strided : !stream.readable<!riscv.freg>):
// CHECK-NEXT:      %a_strided_1 = builtin.unrealized_conversion_cast %a_strided : !stream.readable<!riscv.freg> to !stream.readable<f64>
// CHECK-NEXT:      "test.op"(%a_strided_1) : (!stream.readable<f64>) -> ()
// CHECK-NEXT:    }

// CHECK-NEXT:  }
