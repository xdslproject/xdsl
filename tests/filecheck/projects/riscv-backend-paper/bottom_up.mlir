// RUN: xdsl-opt -p convert-linalg-to-memref-stream,test-optimise-memref-stream,test-lower-memref-stream-to-snitch-stream,test-lower-snitch-stream-to-asm -t riscv-asm %s | filecheck %s


  func.func public @conv_2d_nchw_fchw_d1_s1_3x3(%arg0 : memref<1x1x8x8xf64> {"llvm.noalias"}, %arg1 : memref<1x1x3x3xf64> {"llvm.noalias"}, %arg2 : memref<1x1x6x6xf64> {"llvm.noalias"}) -> memref<1x1x6x6xf64> {
    %cst = arith.constant 0.000000e+00 : f64
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst : f64) outs(%arg2 : memref<1x1x6x6xf64>) {
    ^0(%in : f64, %out : f64):
      linalg.yield %in : f64
    }
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, (d2 + d5), (d3 + d6))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : memref<1x1x8x8xf64>, memref<1x1x3x3xf64>) outs(%arg2 : memref<1x1x6x6xf64>) {
    ^1(%in_1 : f64, %in_2 : f64, %out_1 : f64):
      %0 = arith.mulf %in_1, %in_2 : f64
      %1 = arith.addf %out_1, %0 : f64
      linalg.yield %1 : f64
    }
    func.return %arg2 : memref<1x1x6x6xf64>
  }

// CHECK:       .text
// CHECK-NEXT:  .globl conv_2d_nchw_fchw_d1_s1_3x3
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  conv_2d_nchw_fchw_d1_s1_3x3:
// CHECK-NEXT:      mv t3, a0
// CHECK-NEXT:      mv t2, a1
// CHECK-NEXT:      mv t0, a2
// CHECK-NEXT:      fcvt.d.w ft3, zero
// CHECK-NEXT:      li t1, 5
// CHECK-NEXT:      scfgwi t1, 64
// CHECK-NEXT:      li t1, 2
// CHECK-NEXT:      scfgwi t1, 96
// CHECK-NEXT:      li t1, 2
// CHECK-NEXT:      scfgwi t1, 128
// CHECK-NEXT:      li t1, 5
// CHECK-NEXT:      scfgwi t1, 160
// CHECK-NEXT:      li t1, 8
// CHECK-NEXT:      scfgwi t1, 192
// CHECK-NEXT:      li t1, -32
// CHECK-NEXT:      scfgwi t1, 224
// CHECK-NEXT:      li t1, 8
// CHECK-NEXT:      scfgwi t1, 256
// CHECK-NEXT:      li t1, -120
// CHECK-NEXT:      scfgwi t1, 288
// CHECK-NEXT:      li t1, 5
// CHECK-NEXT:      scfgwi t1, 65
// CHECK-NEXT:      li t1, 2
// CHECK-NEXT:      scfgwi t1, 97
// CHECK-NEXT:      li t1, 2
// CHECK-NEXT:      scfgwi t1, 129
// CHECK-NEXT:      li t1, 5
// CHECK-NEXT:      scfgwi t1, 161
// CHECK-NEXT:      scfgwi zero, 193
// CHECK-NEXT:      li t1, 8
// CHECK-NEXT:      scfgwi t1, 225
// CHECK-NEXT:      li t1, 8
// CHECK-NEXT:      scfgwi t1, 257
// CHECK-NEXT:      li t1, -64
// CHECK-NEXT:      scfgwi t1, 289
// CHECK-NEXT:      li t1, 35
// CHECK-NEXT:      scfgwi t1, 66
// CHECK-NEXT:      li t1, 8
// CHECK-NEXT:      scfgwi t1, 194
// CHECK-NEXT:      scfgwi t3, 864
// CHECK-NEXT:      scfgwi t2, 865
// CHECK-NEXT:      scfgwi t0, 898
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t2, 6
// CHECK-NEXT:      mv t1, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_{{\d}}_for:
// CHECK-NEXT:      fmv.d ft9, ft3
// CHECK-NEXT:      fmv.d ft8, ft3
// CHECK-NEXT:      fmv.d ft7, ft3
// CHECK-NEXT:      fmv.d ft6, ft3
// CHECK-NEXT:      fmv.d ft5, ft3
// CHECK-NEXT:      fmv.d ft4, ft3
// CHECK-NEXT:      li t4, 8
// CHECK-NEXT:      frep.o t4, 12, 0, 0
// CHECK-NEXT:      fmul.d fa3, ft0, ft1
// CHECK-NEXT:      fmul.d fa2, ft0, ft1
// CHECK-NEXT:      fmul.d fa1, ft0, ft1
// CHECK-NEXT:      fmul.d fa0, ft0, ft1
// CHECK-NEXT:      fmul.d ft11, ft0, ft1
// CHECK-NEXT:      fmul.d ft10, ft0, ft1
// CHECK-NEXT:      fadd.d ft9, ft9, fa3
// CHECK-NEXT:      fadd.d ft8, ft8, fa2
// CHECK-NEXT:      fadd.d ft7, ft7, fa1
// CHECK-NEXT:      fadd.d ft6, ft6, fa0
// CHECK-NEXT:      fadd.d ft5, ft5, ft11
// CHECK-NEXT:      fadd.d ft4, ft4, ft10
// CHECK-NEXT:      fmv.d ft2, ft9
// CHECK-NEXT:      fmv.d ft2, ft8
// CHECK-NEXT:      fmv.d ft2, ft7
// CHECK-NEXT:      fmv.d ft2, ft6
// CHECK-NEXT:      fmv.d ft2, ft5
// CHECK-NEXT:      fmv.d ft2, ft4
// CHECK-NEXT:      addi t1, t1, 1
// CHECK-NEXT:      blt t1, t2, scf_body_{{\d}}_for
// CHECK-NEXT:  scf_body_end_{{\d}}_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      mv a0, t0
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
    linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1) -> ()>,
            affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%zero_float : f64) outs(%G : memref<8x8xf64>) {
    ^bb0(%in: f64, %out: f64):
        linalg.yield %in : f64
    }
    linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d2)>,
            affine_map<(d0, d1, d2) -> (d2, d1)>,
            affine_map<(d0, d1, d2) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%X, %Y : memref<8x8xf64>, memref<8x8xf64>) outs(%G : memref<8x8xf64>) {
    ^0(%x : f64, %y : f64, %acc_old : f64):
        %prod = arith.mulf %x, %y fastmath<fast> : f64
        %acc_new = arith.addf %acc_old, %prod fastmath<fast> : f64
        linalg.yield %acc_new : f64
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

  func.func public @pooling_nchw_max_d1_s2_3x3(%arg0 : memref<1x1x10x10xf64> {"llvm.noalias"}, %arg1 : memref<1x1x4x4xf64> {"llvm.noalias"}) -> memref<1x1x4x4xf64> {
    %cst = arith.constant 0.000000e+00 : f64
    memref_stream.generic {
      bounds = [1, 1, 4, 4],
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> ()>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%cst : f64) outs(%arg1 : memref<1x1x4x4xf64>) {
    ^0(%in : f64, %out : f64):
      memref_stream.yield %in : f64
    }
    memref_stream.generic {
      bounds = [1, 1, 4, 4, 3, 3],
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, ((d2 * 2) + d4), ((d3 * 2) + d5))>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
    } ins(%arg0 : memref<1x1x10x10xf64>) outs(%arg1 : memref<1x1x4x4xf64>) {
    ^1(%in_1 : f64, %out_1 : f64):
      %0 = arith.maximumf %out_1, %in_1 fastmath<fast> : f64
      memref_stream.yield %0 : f64
    }
    func.return %arg1 : memref<1x1x4x4xf64>
  }

// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl pooling_nchw_max_d1_s2_3x3
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  pooling_nchw_max_d1_s2_3x3:
// CHECK-NEXT:      mv t2, a0
// CHECK-NEXT:      mv t0, a1
// CHECK-NEXT:      fcvt.d.w ft3, zero
// CHECK-NEXT:      li t3, 3
// CHECK-NEXT:      scfgwi t3, 64
// CHECK-NEXT:      li t3, 2
// CHECK-NEXT:      scfgwi t3, 96
// CHECK-NEXT:      li t3, 2
// CHECK-NEXT:      scfgwi t3, 128
// CHECK-NEXT:      li t3, 3
// CHECK-NEXT:      scfgwi t3, 160
// CHECK-NEXT:      li t3, 16
// CHECK-NEXT:      scfgwi t3, 192
// CHECK-NEXT:      li t3, -40
// CHECK-NEXT:      scfgwi t3, 224
// CHECK-NEXT:      li t3, 16
// CHECK-NEXT:      scfgwi t3, 256
// CHECK-NEXT:      li t3, -64
// CHECK-NEXT:      scfgwi t3, 288
// CHECK-NEXT:      li t3, 15
// CHECK-NEXT:      scfgwi t3, 65
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 193
// CHECK-NEXT:      scfgwi t2, 864
// CHECK-NEXT:      scfgwi t0, 897
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t2, 4
// CHECK-NEXT:      mv t1, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_{{\d}}_for:
// CHECK-NEXT:      fmv.d ft7, ft3
// CHECK-NEXT:      fmv.d ft6, ft3
// CHECK-NEXT:      fmv.d ft5, ft3
// CHECK-NEXT:      fmv.d ft4, ft3
// CHECK-NEXT:      li t4, 8
// CHECK-NEXT:      frep.o t4, 4, 0, 0
// CHECK-NEXT:      fmax.d ft7, ft7, ft0
// CHECK-NEXT:      fmax.d ft6, ft6, ft0
// CHECK-NEXT:      fmax.d ft5, ft5, ft0
// CHECK-NEXT:      fmax.d ft4, ft4, ft0
// CHECK-NEXT:      fmv.d ft1, ft7
// CHECK-NEXT:      fmv.d ft1, ft6
// CHECK-NEXT:      fmv.d ft1, ft5
// CHECK-NEXT:      fmv.d ft1, ft4
// CHECK-NEXT:      addi t1, t1, 1
// CHECK-NEXT:      blt t1, t2, scf_body_{{\d}}_for
// CHECK-NEXT:  scf_body_end_{{\d}}_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      mv a0, t0
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


  func.func public @pooling_nchw_sum_d1_s2_3x3(%arg0 : memref<1x1x10x10xf64> {"llvm.noalias"}, %arg1 : memref<1x1x4x4xf64> {"llvm.noalias"}) -> memref<1x1x4x4xf64> {
    %cst = arith.constant 0.000000e+00 : f64
    memref_stream.generic {
      bounds = [1, 1, 4, 4],
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> ()>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%cst : f64) outs(%arg1 : memref<1x1x4x4xf64>) {
    ^0(%in : f64, %out : f64):
      memref_stream.yield %in : f64
    }
    memref_stream.generic {
      bounds = [1, 1, 4, 4, 3, 3],
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, ((d2 * 2) + d4), ((d3 * 2) + d5))>,
        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
    } ins(%arg0 : memref<1x1x10x10xf64>) outs(%arg1 : memref<1x1x4x4xf64>) {
    ^1(%in_1 : f64, %out_1 : f64):
      %0 = arith.addf %out_1, %in_1 fastmath<fast> : f64
      memref_stream.yield %0 : f64
    }
    func.return %arg1 : memref<1x1x4x4xf64>
  }

// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl pooling_nchw_sum_d1_s2_3x3
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  pooling_nchw_sum_d1_s2_3x3:
// CHECK-NEXT:      mv t2, a0
// CHECK-NEXT:      mv t0, a1
// CHECK-NEXT:      fcvt.d.w ft3, zero
// CHECK-NEXT:      li t3, 3
// CHECK-NEXT:      scfgwi t3, 64
// CHECK-NEXT:      li t3, 2
// CHECK-NEXT:      scfgwi t3, 96
// CHECK-NEXT:      li t3, 2
// CHECK-NEXT:      scfgwi t3, 128
// CHECK-NEXT:      li t3, 3
// CHECK-NEXT:      scfgwi t3, 160
// CHECK-NEXT:      li t3, 16
// CHECK-NEXT:      scfgwi t3, 192
// CHECK-NEXT:      li t3, -40
// CHECK-NEXT:      scfgwi t3, 224
// CHECK-NEXT:      li t3, 16
// CHECK-NEXT:      scfgwi t3, 256
// CHECK-NEXT:      li t3, -64
// CHECK-NEXT:      scfgwi t3, 288
// CHECK-NEXT:      li t3, 15
// CHECK-NEXT:      scfgwi t3, 65
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 193
// CHECK-NEXT:      scfgwi t2, 864
// CHECK-NEXT:      scfgwi t0, 897
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t2, 4
// CHECK-NEXT:      mv t1, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_{{\d}}_for:
// CHECK-NEXT:      fmv.d ft7, ft3
// CHECK-NEXT:      fmv.d ft6, ft3
// CHECK-NEXT:      fmv.d ft5, ft3
// CHECK-NEXT:      fmv.d ft4, ft3
// CHECK-NEXT:      li t4, 8
// CHECK-NEXT:      frep.o t4, 4, 0, 0
// CHECK-NEXT:      fadd.d ft7, ft7, ft0
// CHECK-NEXT:      fadd.d ft6, ft6, ft0
// CHECK-NEXT:      fadd.d ft5, ft5, ft0
// CHECK-NEXT:      fadd.d ft4, ft4, ft0
// CHECK-NEXT:      fmv.d ft1, ft7
// CHECK-NEXT:      fmv.d ft1, ft6
// CHECK-NEXT:      fmv.d ft1, ft5
// CHECK-NEXT:      fmv.d ft1, ft4
// CHECK-NEXT:      addi t1, t1, 1
// CHECK-NEXT:      blt t1, t2, scf_body_{{\d}}_for
// CHECK-NEXT:  scf_body_end_{{\d}}_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      mv a0, t0
// CHECK-NEXT:      ret
