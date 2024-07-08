// RUN: xdsl-opt -p convert-linalg-to-memref-stream,test-optimise-memref-stream,test-lower-memref-stream-to-snitch-stream,test-lower-snitch-stream-to-asm -t riscv-asm %s | filecheck %s

func.func public @conv_2d_nchw_fchw_d1_s1_3x3(
    %X: memref<1x1x8x8xf64>,
    %Y: memref<1x1x3x3xf64>,
    %Z: memref<1x1x6x6xf64>
) -> () {
    %zero_float = arith.constant 0.0 : f64
    linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> ()>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%zero_float : f64) outs(%Z : memref<1x1x6x6xf64>) {
    ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
    }
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
    } ins(%X, %Y : memref<1x1x8x8xf64>, memref<1x1x3x3xf64>) outs(%Z : memref<1x1x6x6xf64>) {
    ^0(%x : f64, %y : f64, %acc : f64):
      %prod = arith.mulf %x, %y fastmath<fast> : f64
      %res = arith.addf %prod, %acc fastmath<fast> : f64
      linalg.yield %res : f64
    }

    func.return
  }


// CHECK:       .text
// CHECK-NEXT:  .globl conv_2d_nchw_fchw_d1_s1_3x3
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  conv_2d_nchw_fchw_d1_s1_3x3:
// CHECK-NEXT:      mv t0, a0
// CHECK-NEXT:      mv t1, a1
// CHECK-NEXT:      mv t2, a2
// CHECK-NEXT:      fcvt.d.w ft3, zero
// CHECK-NEXT:      li t3, 2
// CHECK-NEXT:      scfgwi t3, 64
// CHECK-NEXT:      li t3, 2
// CHECK-NEXT:      scfgwi t3, 96
// CHECK-NEXT:      li t3, 5
// CHECK-NEXT:      scfgwi t3, 128
// CHECK-NEXT:      li t3, 5
// CHECK-NEXT:      scfgwi t3, 160
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 192
// CHECK-NEXT:      li t3, 48
// CHECK-NEXT:      scfgwi t3, 224
// CHECK-NEXT:      li t3, -136
// CHECK-NEXT:      scfgwi t3, 256
// CHECK-NEXT:      li t3, -120
// CHECK-NEXT:      scfgwi t3, 288
// CHECK-NEXT:      li t3, 2
// CHECK-NEXT:      scfgwi t3, 65
// CHECK-NEXT:      li t3, 2
// CHECK-NEXT:      scfgwi t3, 97
// CHECK-NEXT:      li t3, 35
// CHECK-NEXT:      scfgwi t3, 129
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 193
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 225
// CHECK-NEXT:      li t3, -64
// CHECK-NEXT:      scfgwi t3, 257
// CHECK-NEXT:      li t3, 35
// CHECK-NEXT:      scfgwi t3, 66
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 194
// CHECK-NEXT:      scfgwi t0, 864
// CHECK-NEXT:      scfgwi t1, 833
// CHECK-NEXT:      scfgwi t2, 898
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t1, 36
// CHECK-NEXT:      mv t0, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_{{\d+}}_for:
// CHECK-NEXT:      fmv.d ft4, ft3
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      frep.o t3, 1, 0, 0
// CHECK-NEXT:      fmadd.d ft4, ft0, ft1, ft4
// CHECK-NEXT:      fmv.d ft2, ft4
// CHECK-NEXT:      addi t0, t0, 1
// CHECK-NEXT:      blt t0, t1, scf_body_{{\d+}}_for
// CHECK-NEXT:  scf_body_end_{{\d+}}_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret


  func.func @ddot(
    %X : memref<128xf64>,
    %Y : memref<128xf64>,
    %G : memref<f64>
  ) {
    memref_stream.streaming_region {
      patterns = [
          #memref_stream.stride_pattern<ub = [128], index_map = (d0) -> (d0)>,
          #memref_stream.stride_pattern<ub = [128], index_map = (d0) -> (d0)>
      ]
    } ins(%X, %Y : memref<128xf64>, memref<128xf64>) {
    ^0(%x_stream : !stream.readable<f64>, %y_stream : !stream.readable<f64>):
        %zero_float = arith.constant 0.0 : f64

        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %c128 = arith.constant 128 : i32
        %g = scf.for %i = %c0 to %c128 step %c1 iter_args(%acc = %zero_float) -> (f64) {
          %x = memref_stream.read from %x_stream : f64
          %y = memref_stream.read from %y_stream : f64
          %prod = arith.mulf %x, %y fastmath<fast> : f64
          %res = arith.addf %prod, %acc fastmath<fast> : f64
          scf.yield %res : f64
        }

        memref.store %g, %G[] : memref<f64>
    }

    func.return
  }


// CHECK:       .text
// CHECK-NEXT:  .globl ddot
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  ddot:
// CHECK-NEXT:      mv t2, a0
// CHECK-NEXT:      mv t1, a1
// CHECK-NEXT:      mv t0, a2
// CHECK-NEXT:      li t3, 127
// CHECK-NEXT:      scfgwi t3, 95
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 223
// CHECK-NEXT:      scfgwi t2, 768
// CHECK-NEXT:      scfgwi t1, 769
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      fcvt.d.w ft3, zero
// CHECK-NEXT:      li t1, 127
// CHECK-NEXT:      frep.o t1, 1, 0, 0
// CHECK-NEXT:      fmadd.d ft3, ft0, ft1, ft3
// CHECK-NEXT:      fsd ft3, 0(t0)
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret


    func.func @dsum(
      %X : memref<8x16xf64>,
      %Y : memref<8x16xf64>,
      %Z : memref<8x16xf64>
    ) {
      linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
      } ins(%X, %Y : memref<8x16xf64>, memref<8x16xf64>) outs(%Z : memref<8x16xf64>) {
      ^bb0(%x : f64, %y : f64, %out : f64):
          %z = arith.addf %x, %y : f64
          linalg.yield %z : f64
      }

      func.return
    }


// CHECK:       .text
// CHECK-NEXT:  .globl dsum
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  dsum:
// CHECK-NEXT:      mv t2, a0
// CHECK-NEXT:      mv t1, a1
// CHECK-NEXT:      mv t0, a2
// CHECK-NEXT:      li t3, 127
// CHECK-NEXT:      scfgwi t3, 95
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 223
// CHECK-NEXT:      scfgwi t2, 768
// CHECK-NEXT:      scfgwi t1, 769
// CHECK-NEXT:      scfgwi t0, 898
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t0, 127
// CHECK-NEXT:      frep.o t0, 1, 0, 0
// CHECK-NEXT:      fadd.d ft2, ft0, ft1
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret


  // y[ 16 x 16 ]
  func.func @fill(
    %X : f64,
    %Y : memref<16x16xf64>
  ) {
    linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1) -> ()>,
            affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%X : f64) outs(%Y : memref<16x16xf64>) {
    ^bb0(%d : f64, %c : f64):
        linalg.yield %d : f64
    }

    func.return
  }


// CHECK:       .text
// CHECK-NEXT:  .globl fill
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  fill:
// CHECK-NEXT:      fmv.d ft3, fa0
// CHECK-NEXT:      mv t0, a0
// CHECK-NEXT:      li t1, 255
// CHECK-NEXT:      scfgwi t1, 64
// CHECK-NEXT:      li t1, 8
// CHECK-NEXT:      scfgwi t1, 192
// CHECK-NEXT:      scfgwi t0, 896
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t0, 255
// CHECK-NEXT:      frep.o t0, 1, 0, 0
// CHECK-NEXT:      fmv.d ft0, ft3
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret


// x[ M x K ]
// y[ K x N ]
// g[ M x N ]
  func.func @matmul(
    %X : memref<8x8xf64>,
    %Y : memref<8x8xf64>,
    %G : memref<8x8xf64>
  ) {
    %zero_float = arith.constant 0.0 : f64
    memref_stream.streaming_region {
      patterns = [
        #memref_stream.stride_pattern<ub = [8, 2, 8, 4], index_map = (m, n, k, j) -> (m, k)>,
        #memref_stream.stride_pattern<ub = [8, 2, 8, 4], index_map = (m, n, k, j) -> (k, n * 4 + j)>,
        #memref_stream.stride_pattern<ub = [8, 2, 4], index_map = (m, n, j) -> (m, n * 4 + j)>
      ]
    } ins(%X, %Y : memref<8x8xf64>, memref<8x8xf64>) outs(%G : memref<8x8xf64>) {
    ^0(%x_stream : !stream.readable<f64>, %y_stream : !stream.readable<f64>, %g_stream : !stream.writable<f64>):
      memref_stream.generic {
          bounds = [8, 2, 8, 4],
          indexing_maps = [
              affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
              affine_map<(d0, d1, d2, d3) -> (d2, d1 * 4 + d3)>,
              affine_map<(d0, d1, d3) -> (d0, d1 * 4 + d3)>
          ],
          iterator_types = ["parallel", "parallel", "reduction", "interleaved"]
      } ins(%x_stream, %y_stream : !stream.readable<f64>, !stream.readable<f64>) outs(%g_stream : !stream.writable<f64>) inits(%zero_float : f64) {
      ^1(
          %x0 : f64, %x1 : f64, %x2 : f64, %x3 : f64,
          %y0 : f64, %y1 : f64, %y2 : f64, %y3 : f64,
          %g0 : f64, %g1 : f64, %g2 : f64, %g3 : f64
      ):
          %prod0 = arith.mulf %x0, %y0 fastmath<fast> : f64
          %prod1 = arith.mulf %x1, %y1 fastmath<fast> : f64
          %prod2 = arith.mulf %x2, %y2 fastmath<fast> : f64
          %prod3 = arith.mulf %x3, %y3 fastmath<fast> : f64

          %res0 = arith.addf %prod0, %g0 fastmath<fast> : f64
          %res1 = arith.addf %prod1, %g1 fastmath<fast> : f64
          %res2 = arith.addf %prod2, %g2 fastmath<fast> : f64
          %res3 = arith.addf %prod3, %g3 fastmath<fast> : f64

          memref_stream.yield %res0, %res1, %res2, %res3 : f64, f64, f64, f64
      }
    }

    func.return
}

// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  matmul:
// CHECK-NEXT:      mv t0, a0
// CHECK-NEXT:      mv t1, a1
// CHECK-NEXT:      mv t2, a2
// CHECK-NEXT:      fcvt.d.w ft3, zero
// CHECK-NEXT:      li t3, 3
// CHECK-NEXT:      scfgwi t3, 64
// CHECK-NEXT:      li t3, 7
// CHECK-NEXT:      scfgwi t3, 96
// CHECK-NEXT:      li t3, 1
// CHECK-NEXT:      scfgwi t3, 128
// CHECK-NEXT:      li t3, 7
// CHECK-NEXT:      scfgwi t3, 160
// CHECK-NEXT:      scfgwi zero, 192
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 224
// CHECK-NEXT:      li t3, -56
// CHECK-NEXT:      scfgwi t3, 256
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 288
// CHECK-NEXT:      li t3, 3
// CHECK-NEXT:      scfgwi t3, 65
// CHECK-NEXT:      li t3, 7
// CHECK-NEXT:      scfgwi t3, 97
// CHECK-NEXT:      li t3, 1
// CHECK-NEXT:      scfgwi t3, 129
// CHECK-NEXT:      li t3, 7
// CHECK-NEXT:      scfgwi t3, 161
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 193
// CHECK-NEXT:      li t3, 40
// CHECK-NEXT:      scfgwi t3, 225
// CHECK-NEXT:      li t3, -440
// CHECK-NEXT:      scfgwi t3, 257
// CHECK-NEXT:      li t3, -504
// CHECK-NEXT:      scfgwi t3, 289
// CHECK-NEXT:      li t3, 63
// CHECK-NEXT:      scfgwi t3, 66
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 194
// CHECK-NEXT:      scfgwi t0, 864
// CHECK-NEXT:      scfgwi t1, 865
// CHECK-NEXT:      scfgwi t2, 898
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t1, 16
// CHECK-NEXT:      mv t0, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_{{\d+}}_for:
// CHECK-NEXT:      fmv.d ft7, ft3
// CHECK-NEXT:      fmv.d ft6, ft3
// CHECK-NEXT:      fmv.d ft5, ft3
// CHECK-NEXT:      fmv.d ft4, ft3
// CHECK-NEXT:      li t3, 7
// CHECK-NEXT:      frep.o t3, 4, 0, 0
// CHECK-NEXT:      fmadd.d ft7, ft0, ft1, ft7
// CHECK-NEXT:      fmadd.d ft6, ft0, ft1, ft6
// CHECK-NEXT:      fmadd.d ft5, ft0, ft1, ft5
// CHECK-NEXT:      fmadd.d ft4, ft0, ft1, ft4
// CHECK-NEXT:      fmv.d ft2, ft7
// CHECK-NEXT:      fmv.d ft2, ft6
// CHECK-NEXT:      fmv.d ft2, ft5
// CHECK-NEXT:      fmv.d ft2, ft4
// CHECK-NEXT:      addi t0, t0, 1
// CHECK-NEXT:      blt t0, t1, scf_body_{{\d+}}_for
// CHECK-NEXT:  scf_body_end_{{\d+}}_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret

// x[ M x K ]
// y[ K x N ]
// g[ M x N ]
func.func public @pooling_nchw_max_d1_s2_3x3(
    %X: memref<1x1x16x16xf64>,
    %Y: memref<1x1x7x7xf64>
) -> () {
    %min_val = arith.constant -10000 : f64
    linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> ()>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%min_val : f64) outs(%Y : memref<1x1x7x7xf64>) {
    ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
    }
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x3xf32>
    linalg.generic {
      bounds = [#builtin.int<1>, #builtin.int<1>, #builtin.int<7>, #builtin.int<7>, #builtin.int<3>, #builtin.int<3>],
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d3 * 2 + d5)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
    } ins(%X, %alloc : memref<1x1x16x16xf64>, memref<3x3xf32>) outs(%Y : memref<1x1x7x7xf64>) {
    ^0(%x : f64, %alloc_val: f64, %acc : f64):
      %res = arith.maximumf %x, %acc : f64
      linalg.yield %res : f64
    }
    memref.dealloc %alloc : memref<3x3xf32>
    func.return
  }


// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl pooling_nchw_max_d1_s2_3x3
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  pooling_nchw_max_d1_s2_3x3:
// CHECK-NEXT:      mv t1, a0
// CHECK-NEXT:      mv t2, a1
// CHECK-NEXT:      li t0, -10000
// CHECK-NEXT:      fcvt.d.w ft3, t0
// CHECK-NEXT:      li t0, 2
// CHECK-NEXT:      scfgwi t0, 64
// CHECK-NEXT:      li t0, 2
// CHECK-NEXT:      scfgwi t0, 96
// CHECK-NEXT:      li t0, 6
// CHECK-NEXT:      scfgwi t0, 128
// CHECK-NEXT:      li t0, 6
// CHECK-NEXT:      scfgwi t0, 160
// CHECK-NEXT:      li t0, 8
// CHECK-NEXT:      scfgwi t0, 192
// CHECK-NEXT:      li t0, 112
// CHECK-NEXT:      scfgwi t0, 224
// CHECK-NEXT:      li t0, -256
// CHECK-NEXT:      scfgwi t0, 256
// CHECK-NEXT:      li t0, -112
// CHECK-NEXT:      scfgwi t0, 288
// CHECK-NEXT:      li t0, 48
// CHECK-NEXT:      scfgwi t0, 65
// CHECK-NEXT:      li t0, 8
// CHECK-NEXT:      scfgwi t0, 193
// CHECK-NEXT:      scfgwi t1, 864
// CHECK-NEXT:      scfgwi t2, 897
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t1, 49
// CHECK-NEXT:      mv t0, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_{{\d+}}_for:
// CHECK-NEXT:      fmv.d ft4, ft3
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      frep.o t3, 1, 0, 0
// CHECK-NEXT:      fmax.d ft4, ft0, ft4
// CHECK-NEXT:      fmv.d ft1, ft4
// CHECK-NEXT:      addi t0, t0, 1
// CHECK-NEXT:      blt t0, t1, scf_body_{{\d+}}_for
// CHECK-NEXT:  scf_body_end_{{\d+}}_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret


  func.func public @relu(%X: memref<16x16xf64>, %Y: memref<16x16xf64>) {
    %zero_float = arith.constant 0.0 : f64

    linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%X : memref<16x16xf64>) outs(%Y : memref<16x16xf64>) {
    ^bb0(%x : f64, %out : f64):
        %y = arith.maximumf %x, %zero_float : f64
        linalg.yield %y : f64
    }

    func.return
  }


// CHECK:       .text
// CHECK-NEXT:  .globl relu
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  relu:
// CHECK-NEXT:      mv t1, a0
// CHECK-NEXT:      mv t0, a1
// CHECK-NEXT:      fcvt.d.w ft3, zero
// CHECK-NEXT:      li t2, 255
// CHECK-NEXT:      scfgwi t2, 95
// CHECK-NEXT:      li t2, 8
// CHECK-NEXT:      scfgwi t2, 223
// CHECK-NEXT:      scfgwi t1, 768
// CHECK-NEXT:      scfgwi t0, 897
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t0, 255
// CHECK-NEXT:      frep.o t0, 1, 0, 0
// CHECK-NEXT:      fmax.d ft1, ft0, ft3
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret


// x[ M x K ]
// y[ K x N ]
// g[ M x N ]
func.func public @pooling_nchw_sum_d1_s2_3x3(
    %X: memref<1x1x16x16xf64>,
    %Y: memref<1x1x7x7xf64>
) -> () {
    %zero_float = arith.constant 0.0 : f64
    linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> ()>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%zero_float : f64) outs(%Y : memref<1x1x7x7xf64>) {
    ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
    }
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x3xf32>
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d3 * 2 + d5)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
    } ins(%X, %alloc : memref<1x1x16x16xf64>, memref<3x3xf32>) outs(%Y : memref<1x1x7x7xf64>) {
    ^0(%x : f64, %alloc_val: f64, %acc : f64):
      %res = arith.addf %x, %acc : f64
      linalg.yield %res : f64
    }
    memref.dealloc %alloc : memref<3x3xf32>
    func.return
  }


// CHECK:       .text
// CHECK-NEXT:  .globl pooling_nchw_sum_d1_s2_3x3
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  pooling_nchw_sum_d1_s2_3x3:
// CHECK-NEXT:      mv t1, a0
// CHECK-NEXT:      mv t2, a1
// CHECK-NEXT:      fcvt.d.w ft3, zero
// CHECK-NEXT:      li t0, 2
// CHECK-NEXT:      scfgwi t0, 64
// CHECK-NEXT:      li t0, 2
// CHECK-NEXT:      scfgwi t0, 96
// CHECK-NEXT:      li t0, 6
// CHECK-NEXT:      scfgwi t0, 128
// CHECK-NEXT:      li t0, 6
// CHECK-NEXT:      scfgwi t0, 160
// CHECK-NEXT:      li t0, 8
// CHECK-NEXT:      scfgwi t0, 192
// CHECK-NEXT:      li t0, 112
// CHECK-NEXT:      scfgwi t0, 224
// CHECK-NEXT:      li t0, -256
// CHECK-NEXT:      scfgwi t0, 256
// CHECK-NEXT:      li t0, -112
// CHECK-NEXT:      scfgwi t0, 288
// CHECK-NEXT:      li t0, 48
// CHECK-NEXT:      scfgwi t0, 65
// CHECK-NEXT:      li t0, 8
// CHECK-NEXT:      scfgwi t0, 193
// CHECK-NEXT:      scfgwi t1, 864
// CHECK-NEXT:      scfgwi t2, 897
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t1, 49
// CHECK-NEXT:      mv t0, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_{{\d+}}_for:
// CHECK-NEXT:      fmv.d ft4, ft3
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      frep.o t3, 1, 0, 0
// CHECK-NEXT:      fadd.d ft4, ft0, ft4
// CHECK-NEXT:      fmv.d ft1, ft4
// CHECK-NEXT:      addi t0, t0, 1
// CHECK-NEXT:      blt t0, t1, scf_body_{{\d+}}_for
// CHECK-NEXT:  scf_body_end_{{\d+}}_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret
