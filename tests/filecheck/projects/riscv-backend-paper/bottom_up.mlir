// RUN: xdsl-opt -p convert-linalg-to-memref-stream,test-optimise-memref-stream,test-lower-memref-stream-to-snitch-stream,test-lower-snitch-stream-to-asm -t riscv-asm %s | filecheck %s


func.func public @ssum(
  %X: memref<8x16xf32>,
  %Y: memref<8x16xf32>,
  %Z: memref<8x16xf32>
) {
  %X_1 = builtin.unrealized_conversion_cast %X : memref<8x16xf32> to !riscv.reg
  %Y_1 = builtin.unrealized_conversion_cast %Y : memref<8x16xf32> to !riscv.reg
  %Z_1 = builtin.unrealized_conversion_cast %Z : memref<8x16xf32> to !riscv.reg
  snitch_stream.streaming_region {
    patterns = [
      #snitch_stream.stride_pattern<ub = [64], strides = [8]>
    ]
  } ins(%X_1, %Y_1 : !riscv.reg, !riscv.reg) outs(%Z_1 : !riscv.reg) {
  ^0(%x : !stream.readable<!riscv.freg>, %y : !stream.readable<!riscv.freg>, %0 : !stream.writable<!riscv.freg>):
    %1 = riscv.li 8 : !riscv.reg
    %2 = riscv.li 0 : !riscv.reg
    %3 = riscv.li 1 : !riscv.reg
    %4 = riscv.li 64 : !riscv.reg
    riscv_scf.for %5 : !riscv.reg = %2 to %4 step %3 {
      %x_1 = riscv_snitch.read from %x : !riscv.freg
      %y_1 = riscv_snitch.read from %y : !riscv.freg
      %z = riscv.vfadd.s %x_1, %y_1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
      riscv_snitch.write %z to %0 : !riscv.freg
    }
  }
  func.return
}

// CHECK:       .text
// CHECK-NEXT:  .globl ssum
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  ssum:
// CHECK-NEXT:      mv t2, a0
// CHECK-NEXT:      mv t1, a1
// CHECK-NEXT:      mv t0, a2
// CHECK-NEXT:      li t3, 63
// CHECK-NEXT:      scfgwi t3, 95                                # dm 31 dim 0 bound
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 223                               # dm 31 dim 0 stride
// CHECK-NEXT:      scfgwi zero, 63                              # dm 31 repeat
// CHECK-NEXT:      scfgwi t2, 768                               # dm 0 dim 0 source
// CHECK-NEXT:      scfgwi t1, 769                               # dm 1 dim 0 source
// CHECK-NEXT:      scfgwi t0, 898                               # dm 2 dim 0 destination
// CHECK-NEXT:      csrrsi zero, 1984, 1                         # SSR enable
// CHECK-NEXT:      li t0, 63
// CHECK-NEXT:      frep.o t0, 1, 0, 0
// CHECK-NEXT:      vfadd.s ft2, ft0, ft1
// CHECK-NEXT:      csrrci zero, 1984, 1                         # SSR disable
// CHECK-NEXT:      ret

func.func public @conv_2d_nchw_fchw_d1_s1_3x3(
    %X: memref<1x1x10x10xf64>,
    %Y: memref<1x1x3x3xf64>,
    %Z: memref<1x1x8x8xf64>
) -> () {
    %zero_float = arith.constant 0.0 : f64
    linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> ()>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%zero_float : f64) outs(%Z : memref<1x1x8x8xf64>) {
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
    } ins(%X, %Y : memref<1x1x10x10xf64>, memref<1x1x3x3xf64>) outs(%Z : memref<1x1x8x8xf64>) {
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
// CHECK-NEXT:      mv t2, a0
// CHECK-NEXT:      mv t1, a1
// CHECK-NEXT:      mv t0, a2
// CHECK-NEXT:      fcvt.d.w ft3, zero
// CHECK-NEXT:      li t4, 8
// CHECK-NEXT:      mv t3, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_{{\d}}_for:
// CHECK-NEXT:      mv t6, t3
// CHECK-NEXT:      li a4, 10
// CHECK-NEXT:      mul t6, t6, a4
// CHECK-NEXT:      li a4, 8
// CHECK-NEXT:      mul t6, t6, a4                               # multiply by element size
// CHECK-NEXT:      add t6, t2, t6
// CHECK-NEXT:      mv a4, t1
// CHECK-NEXT:      li a3, 8
// CHECK-NEXT:      mul a3, t3, a3
// CHECK-NEXT:      li a5, 8
// CHECK-NEXT:      mul a3, a3, a5                               # multiply by element size
// CHECK-NEXT:      add a3, t0, a3
// CHECK-NEXT:      li a5, 3
// CHECK-NEXT:      scfgwi a5, 64                                # dm 0 dim 0 bound
// CHECK-NEXT:      li a5, 2
// CHECK-NEXT:      scfgwi a5, 96                                # dm 0 dim 1 bound
// CHECK-NEXT:      li a5, 2
// CHECK-NEXT:      scfgwi a5, 128                               # dm 0 dim 2 bound
// CHECK-NEXT:      li a5, 1
// CHECK-NEXT:      scfgwi a5, 160                               # dm 0 dim 3 bound
// CHECK-NEXT:      li a5, 8
// CHECK-NEXT:      scfgwi a5, 192                               # dm 0 dim 0 stride
// CHECK-NEXT:      li a5, -16
// CHECK-NEXT:      scfgwi a5, 224                               # dm 0 dim 1 stride
// CHECK-NEXT:      li a5, 40
// CHECK-NEXT:      scfgwi a5, 256                               # dm 0 dim 2 stride
// CHECK-NEXT:      li a5, -168
// CHECK-NEXT:      scfgwi a5, 288                               # dm 0 dim 3 stride
// CHECK-NEXT:      scfgwi zero, 32                              # dm 0 repeat
// CHECK-NEXT:      li a5, 2
// CHECK-NEXT:      scfgwi a5, 65                                # dm 1 dim 0 bound
// CHECK-NEXT:      li a5, 2
// CHECK-NEXT:      scfgwi a5, 97                                # dm 1 dim 1 bound
// CHECK-NEXT:      li a5, 1
// CHECK-NEXT:      scfgwi a5, 129                               # dm 1 dim 2 bound
// CHECK-NEXT:      li a5, 8
// CHECK-NEXT:      scfgwi a5, 193                               # dm 1 dim 0 stride
// CHECK-NEXT:      li a5, 8
// CHECK-NEXT:      scfgwi a5, 225                               # dm 1 dim 1 stride
// CHECK-NEXT:      li a5, -64
// CHECK-NEXT:      scfgwi a5, 257                               # dm 1 dim 2 stride
// CHECK-NEXT:      li a5, 3
// CHECK-NEXT:      scfgwi a5, 33                                # dm 1 repeat
// CHECK-NEXT:      li a5, 7
// CHECK-NEXT:      scfgwi a5, 66                                # dm 2 dim 0 bound
// CHECK-NEXT:      li a5, 8
// CHECK-NEXT:      scfgwi a5, 194                               # dm 2 dim 0 stride
// CHECK-NEXT:      scfgwi zero, 34                              # dm 2 repeat
// CHECK-NEXT:      scfgwi t6, 864                               # dm 0 dim 3 source
// CHECK-NEXT:      scfgwi a4, 833                               # dm 1 dim 2 source
// CHECK-NEXT:      scfgwi a3, 898                               # dm 2 dim 0 destination
// CHECK-NEXT:      csrrsi zero, 1984, 1                         # SSR enable
// CHECK-NEXT:      li a3, 2
// CHECK-NEXT:      mv t6, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_{{\d}}_for:
// CHECK-NEXT:      fmv.d ft7, ft3
// CHECK-NEXT:      fmv.d ft6, ft3
// CHECK-NEXT:      fmv.d ft5, ft3
// CHECK-NEXT:      fmv.d ft4, ft3
// CHECK-NEXT:      li a5, 8
// CHECK-NEXT:      frep.o a5, 4, 0, 0
// CHECK-NEXT:      fmadd.d ft7, ft0, ft1, ft7
// CHECK-NEXT:      fmadd.d ft6, ft0, ft1, ft6
// CHECK-NEXT:      fmadd.d ft5, ft0, ft1, ft5
// CHECK-NEXT:      fmadd.d ft4, ft0, ft1, ft4
// CHECK-NEXT:      fmv.d ft2, ft7
// CHECK-NEXT:      fmv.d ft2, ft6
// CHECK-NEXT:      fmv.d ft2, ft5
// CHECK-NEXT:      fmv.d ft2, ft4
// CHECK-NEXT:      addi t6, t6, 1
// CHECK-NEXT:      blt t6, a3, scf_body_{{\d}}_for
// CHECK-NEXT:  scf_body_end_{{\d}}_for:
// CHECK-NEXT:      csrrci zero, 1984, 1                         # SSR disable
// CHECK-NEXT:      addi t3, t3, 1
// CHECK-NEXT:      blt t3, t4, scf_body_{{\d}}_for
// CHECK-NEXT:  scf_body_end_{{\d}}_for:
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
// CHECK-NEXT:      scfgwi t3, 95                                # dm 31 dim 0 bound
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 223                               # dm 31 dim 0 stride
// CHECK-NEXT:      scfgwi zero, 63                              # dm 31 repeat
// CHECK-NEXT:      scfgwi t2, 768                               # dm 0 dim 0 source
// CHECK-NEXT:      scfgwi t1, 769                               # dm 1 dim 0 source
// CHECK-NEXT:      csrrsi zero, 1984, 1                         # SSR enable
// CHECK-NEXT:      fcvt.d.w ft3, zero
// CHECK-NEXT:      li t1, 127
// CHECK-NEXT:      frep.o t1, 1, 0, 0
// CHECK-NEXT:      fmadd.d ft3, ft0, ft1, ft3
// CHECK-NEXT:      fsd ft3, 0(t0)                               # store double value to memref of shape ()
// CHECK-NEXT:      csrrci zero, 1984, 1                         # SSR disable
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
// CHECK-NEXT:      scfgwi t3, 95                                # dm 31 dim 0 bound
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 223                               # dm 31 dim 0 stride
// CHECK-NEXT:      scfgwi zero, 63                              # dm 31 repeat
// CHECK-NEXT:      scfgwi t2, 768                               # dm 0 dim 0 source
// CHECK-NEXT:      scfgwi t1, 769                               # dm 1 dim 0 source
// CHECK-NEXT:      scfgwi t0, 898                               # dm 2 dim 0 destination
// CHECK-NEXT:      csrrsi zero, 1984, 1                         # SSR enable
// CHECK-NEXT:      li t0, 127
// CHECK-NEXT:      frep.o t0, 1, 0, 0
// CHECK-NEXT:      fadd.d ft2, ft0, ft1
// CHECK-NEXT:      csrrci zero, 1984, 1                         # SSR disable
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
// CHECK-NEXT:      scfgwi t1, 64                                # dm 0 dim 0 bound
// CHECK-NEXT:      li t1, 8
// CHECK-NEXT:      scfgwi t1, 192                               # dm 0 dim 0 stride
// CHECK-NEXT:      scfgwi zero, 32                              # dm 0 repeat
// CHECK-NEXT:      scfgwi t0, 896                               # dm 0 dim 0 destination
// CHECK-NEXT:      csrrsi zero, 1984, 1                         # SSR enable
// CHECK-NEXT:      li t0, 255
// CHECK-NEXT:      frep.o t0, 1, 0, 0
// CHECK-NEXT:      fmv.d ft0, ft3
// CHECK-NEXT:      csrrci zero, 1984, 1                         # SSR disable
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
// CHECK-NEXT:      li t3, 7
// CHECK-NEXT:      scfgwi t3, 64                                # dm 0 dim 0 bound
// CHECK-NEXT:      li t3, 1
// CHECK-NEXT:      scfgwi t3, 96                                # dm 0 dim 1 bound
// CHECK-NEXT:      li t3, 7
// CHECK-NEXT:      scfgwi t3, 128                               # dm 0 dim 2 bound
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 192                               # dm 0 dim 0 stride
// CHECK-NEXT:      li t3, -56
// CHECK-NEXT:      scfgwi t3, 224                               # dm 0 dim 1 stride
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 256                               # dm 0 dim 2 stride
// CHECK-NEXT:      li t3, 3
// CHECK-NEXT:      scfgwi t3, 32                                # dm 0 repeat
// CHECK-NEXT:      li t3, 3
// CHECK-NEXT:      scfgwi t3, 65                                # dm 1 dim 0 bound
// CHECK-NEXT:      li t3, 7
// CHECK-NEXT:      scfgwi t3, 97                                # dm 1 dim 1 bound
// CHECK-NEXT:      li t3, 1
// CHECK-NEXT:      scfgwi t3, 129                               # dm 1 dim 2 bound
// CHECK-NEXT:      li t3, 7
// CHECK-NEXT:      scfgwi t3, 161                               # dm 1 dim 3 bound
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 193                               # dm 1 dim 0 stride
// CHECK-NEXT:      li t3, 40
// CHECK-NEXT:      scfgwi t3, 225                               # dm 1 dim 1 stride
// CHECK-NEXT:      li t3, -440
// CHECK-NEXT:      scfgwi t3, 257                               # dm 1 dim 2 stride
// CHECK-NEXT:      li t3, -504
// CHECK-NEXT:      scfgwi t3, 289                               # dm 1 dim 3 stride
// CHECK-NEXT:      scfgwi zero, 33                              # dm 1 repeat
// CHECK-NEXT:      li t3, 63
// CHECK-NEXT:      scfgwi t3, 66                                # dm 2 dim 0 bound
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      scfgwi t3, 194                               # dm 2 dim 0 stride
// CHECK-NEXT:      scfgwi zero, 34                              # dm 2 repeat
// CHECK-NEXT:      scfgwi t0, 832                               # dm 0 dim 2 source
// CHECK-NEXT:      scfgwi t1, 865                               # dm 1 dim 3 source
// CHECK-NEXT:      scfgwi t2, 898                               # dm 2 dim 0 destination
// CHECK-NEXT:      csrrsi zero, 1984, 1                         # SSR enable
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
// CHECK-NEXT:      csrrci zero, 1984, 1                         # SSR disable
// CHECK-NEXT:      ret

// x[ M x K ]
// y[ K x N ]
// g[ M x N ]
func.func public @pooling_nchw_max_d1_s2_3x3(
    %X: memref<1x1x18x18xf64>,
    %Y: memref<1x1x8x8xf64>
) -> () {
    %min_val = arith.constant -10000 : f64
    linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> ()>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%min_val : f64) outs(%Y : memref<1x1x8x8xf64>) {
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
    } ins(%X, %alloc : memref<1x1x18x18xf64>, memref<3x3xf32>) outs(%Y : memref<1x1x8x8xf64>) {
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
// CHECK-NEXT:      mv t0, a1
// CHECK-NEXT:      li t4, -10000
// CHECK-NEXT:      fcvt.d.w ft3, t4
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      mv t2, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_{{\d}}_for:
// CHECK-NEXT:      li a2, 2
// CHECK-NEXT:      mul a2, t2, a2
// CHECK-NEXT:      li t6, 18
// CHECK-NEXT:      mul a2, a2, t6
// CHECK-NEXT:      li t6, 8
// CHECK-NEXT:      mul a2, a2, t6                               # multiply by element size
// CHECK-NEXT:      add a2, t1, a2
// CHECK-NEXT:      li t6, 8
// CHECK-NEXT:      mul t6, t2, t6
// CHECK-NEXT:      li t5, 8
// CHECK-NEXT:      mul t6, t6, t5                               # multiply by element size
// CHECK-NEXT:      add t6, t0, t6
// CHECK-NEXT:      li t5, 3
// CHECK-NEXT:      scfgwi t5, 64                                # dm 0 dim 0 bound
// CHECK-NEXT:      li t5, 2
// CHECK-NEXT:      scfgwi t5, 96                                # dm 0 dim 1 bound
// CHECK-NEXT:      li t5, 2
// CHECK-NEXT:      scfgwi t5, 128                               # dm 0 dim 2 bound
// CHECK-NEXT:      li t5, 1
// CHECK-NEXT:      scfgwi t5, 160                               # dm 0 dim 3 bound
// CHECK-NEXT:      li t5, 16
// CHECK-NEXT:      scfgwi t5, 192                               # dm 0 dim 0 stride
// CHECK-NEXT:      li t5, -40
// CHECK-NEXT:      scfgwi t5, 224                               # dm 0 dim 1 stride
// CHECK-NEXT:      li t5, 80
// CHECK-NEXT:      scfgwi t5, 256                               # dm 0 dim 2 stride
// CHECK-NEXT:      li t5, -288
// CHECK-NEXT:      scfgwi t5, 288                               # dm 0 dim 3 stride
// CHECK-NEXT:      scfgwi zero, 32                              # dm 0 repeat
// CHECK-NEXT:      li t5, 7
// CHECK-NEXT:      scfgwi t5, 65                                # dm 1 dim 0 bound
// CHECK-NEXT:      li t5, 8
// CHECK-NEXT:      scfgwi t5, 193                               # dm 1 dim 0 stride
// CHECK-NEXT:      scfgwi zero, 33                              # dm 1 repeat
// CHECK-NEXT:      scfgwi a2, 864                               # dm 0 dim 3 source
// CHECK-NEXT:      scfgwi t6, 897                               # dm 1 dim 0 destination
// CHECK-NEXT:      csrrsi zero, 1984, 1                         # SSR enable
// CHECK-NEXT:      li t6, 2
// CHECK-NEXT:      mv t5, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_{{\d}}_for:
// CHECK-NEXT:      fmv.d ft7, ft3
// CHECK-NEXT:      fmv.d ft6, ft3
// CHECK-NEXT:      fmv.d ft5, ft3
// CHECK-NEXT:      fmv.d ft4, ft3
// CHECK-NEXT:      li a3, 8
// CHECK-NEXT:      frep.o a3, 4, 0, 0
// CHECK-NEXT:      fmax.d ft7, ft0, ft7
// CHECK-NEXT:      fmax.d ft6, ft0, ft6
// CHECK-NEXT:      fmax.d ft5, ft0, ft5
// CHECK-NEXT:      fmax.d ft4, ft0, ft4
// CHECK-NEXT:      fmv.d ft1, ft7
// CHECK-NEXT:      fmv.d ft1, ft6
// CHECK-NEXT:      fmv.d ft1, ft5
// CHECK-NEXT:      fmv.d ft1, ft4
// CHECK-NEXT:      addi t5, t5, 1
// CHECK-NEXT:      blt t5, t6, scf_body_{{\d}}_for
// CHECK-NEXT:  scf_body_end_{{\d}}_for:
// CHECK-NEXT:      csrrci zero, 1984, 1                         # SSR disable
// CHECK-NEXT:      addi t2, t2, 1
// CHECK-NEXT:      blt t2, t3, scf_body_{{\d}}_for
// CHECK-NEXT:  scf_body_end_{{\d}}_for:
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
// CHECK-NEXT:      scfgwi t2, 95                                # dm 31 dim 0 bound
// CHECK-NEXT:      li t2, 8
// CHECK-NEXT:      scfgwi t2, 223                               # dm 31 dim 0 stride
// CHECK-NEXT:      scfgwi zero, 63                              # dm 31 repeat
// CHECK-NEXT:      scfgwi t1, 768                               # dm 0 dim 0 source
// CHECK-NEXT:      scfgwi t0, 897                               # dm 1 dim 0 destination
// CHECK-NEXT:      csrrsi zero, 1984, 1                         # SSR enable
// CHECK-NEXT:      li t0, 255
// CHECK-NEXT:      frep.o t0, 1, 0, 0
// CHECK-NEXT:      fmax.d ft1, ft0, ft3
// CHECK-NEXT:      csrrci zero, 1984, 1                         # SSR disable
// CHECK-NEXT:      ret


// x[ M x K ]
// y[ K x N ]
// g[ M x N ]
func.func public @pooling_nchw_sum_d1_s2_3x3(
    %X: memref<1x1x18x18xf64>,
    %Y: memref<1x1x8x8xf64>
) -> () {
    %zero_float = arith.constant 0.0 : f64
    linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> ()>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%zero_float : f64) outs(%Y : memref<1x1x8x8xf64>) {
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
    } ins(%X, %alloc : memref<1x1x18x18xf64>, memref<3x3xf32>) outs(%Y : memref<1x1x8x8xf64>) {
    ^0(%x : f64, %alloc_val: f64, %acc : f64):
      %res = arith.addf %x, %acc : f64
      linalg.yield %res : f64
    }
    memref.dealloc %alloc : memref<3x3xf32>
    func.return
  }


// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl pooling_nchw_sum_d1_s2_3x3
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  pooling_nchw_sum_d1_s2_3x3:
// CHECK-NEXT:      mv t1, a0
// CHECK-NEXT:      mv t0, a1
// CHECK-NEXT:      fcvt.d.w ft3, zero
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      mv t2, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_{{\d}}_for:
// CHECK-NEXT:      li a2, 2
// CHECK-NEXT:      mul a2, t2, a2
// CHECK-NEXT:      li t6, 18
// CHECK-NEXT:      mul a2, a2, t6
// CHECK-NEXT:      li t6, 8
// CHECK-NEXT:      mul a2, a2, t6                               # multiply by element size
// CHECK-NEXT:      add a2, t1, a2
// CHECK-NEXT:      li t6, 8
// CHECK-NEXT:      mul t6, t2, t6
// CHECK-NEXT:      li t5, 8
// CHECK-NEXT:      mul t6, t6, t5                               # multiply by element size
// CHECK-NEXT:      add t6, t0, t6
// CHECK-NEXT:      li t5, 3
// CHECK-NEXT:      scfgwi t5, 64                                # dm 0 dim 0 bound
// CHECK-NEXT:      li t5, 2
// CHECK-NEXT:      scfgwi t5, 96                                # dm 0 dim 1 bound
// CHECK-NEXT:      li t5, 2
// CHECK-NEXT:      scfgwi t5, 128                               # dm 0 dim 2 bound
// CHECK-NEXT:      li t5, 1
// CHECK-NEXT:      scfgwi t5, 160                               # dm 0 dim 3 bound
// CHECK-NEXT:      li t5, 16
// CHECK-NEXT:      scfgwi t5, 192                               # dm 0 dim 0 stride
// CHECK-NEXT:      li t5, -40
// CHECK-NEXT:      scfgwi t5, 224                               # dm 0 dim 1 stride
// CHECK-NEXT:      li t5, 80
// CHECK-NEXT:      scfgwi t5, 256                               # dm 0 dim 2 stride
// CHECK-NEXT:      li t5, -288
// CHECK-NEXT:      scfgwi t5, 288                               # dm 0 dim 3 stride
// CHECK-NEXT:      scfgwi zero, 32                              # dm 0 repeat
// CHECK-NEXT:      li t5, 7
// CHECK-NEXT:      scfgwi t5, 65                                # dm 1 dim 0 bound
// CHECK-NEXT:      li t5, 8
// CHECK-NEXT:      scfgwi t5, 193                               # dm 1 dim 0 stride
// CHECK-NEXT:      scfgwi zero, 33                              # dm 1 repeat
// CHECK-NEXT:      scfgwi a2, 864                               # dm 0 dim 3 source
// CHECK-NEXT:      scfgwi t6, 897                               # dm 1 dim 0 destination
// CHECK-NEXT:      csrrsi zero, 1984, 1                         # SSR enable
// CHECK-NEXT:      li t6, 2
// CHECK-NEXT:      mv t5, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_{{\d}}_for:
// CHECK-NEXT:      fmv.d ft7, ft3
// CHECK-NEXT:      fmv.d ft6, ft3
// CHECK-NEXT:      fmv.d ft5, ft3
// CHECK-NEXT:      fmv.d ft4, ft3
// CHECK-NEXT:      li a3, 8
// CHECK-NEXT:      frep.o a3, 4, 0, 0
// CHECK-NEXT:      fadd.d ft7, ft0, ft7
// CHECK-NEXT:      fadd.d ft6, ft0, ft6
// CHECK-NEXT:      fadd.d ft5, ft0, ft5
// CHECK-NEXT:      fadd.d ft4, ft0, ft4
// CHECK-NEXT:      fmv.d ft1, ft7
// CHECK-NEXT:      fmv.d ft1, ft6
// CHECK-NEXT:      fmv.d ft1, ft5
// CHECK-NEXT:      fmv.d ft1, ft4
// CHECK-NEXT:      addi t5, t5, 1
// CHECK-NEXT:      blt t5, t6, scf_body_{{\d}}_for
// CHECK-NEXT:  scf_body_end_{{\d}}_for:
// CHECK-NEXT:      csrrci zero, 1984, 1                         # SSR disable
// CHECK-NEXT:      addi t2, t2, 1
// CHECK-NEXT:      blt t2, t3, scf_body_{{\d}}_for
// CHECK-NEXT:  scf_body_end_{{\d}}_for:
// CHECK-NEXT:      ret
