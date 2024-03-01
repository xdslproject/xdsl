// RUN: xdsl-opt -p convert-func-to-riscv-func,convert-memref-stream-to-snitch,reconcile-unrealized-casts,test-lower-snitch-stream-to-asm -t riscv-asm %s | filecheck %s

// x[ M x K ]
// y[ K x N ]
// g[ M x N ]
func.func public @conv_2d_nchw_fchw_d1_s1_3x3(
    %X: memref<1x1x8x8xf64>,
    %Y: memref<1x1x3x3xf64>,
    %Z: memref<1x1x6x6xf64>
) -> () {
    %X_moved = builtin.unrealized_conversion_cast %X : memref<1x1x8x8xf64> to !riscv.reg<>
    %Y_moved = builtin.unrealized_conversion_cast %Y : memref<1x1x3x3xf64> to !riscv.reg<>
    %Z_moved = builtin.unrealized_conversion_cast %Z : memref<1x1x6x6xf64> to !riscv.reg<>

    %c0 = riscv.li 0 : () -> !riscv.reg<>
    %c1 = riscv.li 1 : () -> !riscv.reg<>
    %c8 = riscv.li 8 : () -> !riscv.reg<>
    %c9 = riscv.li 9 : () -> !riscv.reg<>

    %zero_float = riscv.fcvt.d.w %c0 : (!riscv.reg<>) -> !riscv.freg<>

    memref_stream.streaming_region {
      bounds = [1, 1, 6, 6, 1, 3, 3],
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
      ]
    } ins(%X, %Y : memref<1x1x8x8xf64>, memref<1x1x3x3xf64>) {
    ^0(%x_stream : !stream.readable<f64>, %y_stream : !stream.readable<f64>):
      %X_stream = builtin.unrealized_conversion_cast %x_stream : !stream.readable<f64> to !stream.readable<!riscv.freg<>>
      %Y_stream = builtin.unrealized_conversion_cast %y_stream : !stream.readable<f64> to !stream.readable<!riscv.freg<>>

      %c288 = riscv.li 288 : () -> !riscv.reg<>
      riscv_scf.for %z_i : !riscv.reg<> = %c0 to %c288 step %c8 {
        %Z_dest = riscv.add %Z_moved, %z_i : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %c = riscv.fld %Z_dest, 0 : (!riscv.reg<>) -> !riscv.freg<>

        %z = riscv_scf.for %i : !riscv.reg<> = %c0 to %c9 step %c1 iter_args(%acc = %c) -> (!riscv.freg<>) {
          %x = riscv_snitch.read from %X_stream : !riscv.freg<>
          %y = riscv_snitch.read from %Y_stream : !riscv.freg<>
          %res = riscv.fmadd.d %x, %y, %acc : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
          riscv_scf.yield %res : !riscv.freg<>
        }

        riscv.fsd %Z_dest, %z, 0 : (!riscv.reg<>, !riscv.freg<>) -> ()

        riscv_scf.yield
      }
    }

    func.return
  }


// CHECK:       .text
// CHECK-NEXT:  .globl conv_2d_nchw_fchw_d1_s1_3x3
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  conv_2d_nchw_fchw_d1_s1_3x3:
// CHECK-NEXT:      mv t4, a0
// CHECK-NEXT:      mv t2, a1
// CHECK-NEXT:      mv t0, a2
// CHECK-NEXT:      li t5, 8
// CHECK-NEXT:      li a5, 2
// CHECK-NEXT:      li t6, 2
// CHECK-NEXT:      li a3, 5
// CHECK-NEXT:      li a4, 5
// CHECK-NEXT:      scfgwi a5, 64
// CHECK-NEXT:      scfgwi t6, 96
// CHECK-NEXT:      scfgwi a3, 128
// CHECK-NEXT:      scfgwi a4, 160
// CHECK-NEXT:      scfgwi t5, 192
// CHECK-NEXT:      li t5, 48
// CHECK-NEXT:      scfgwi t5, 224
// CHECK-NEXT:      li t5, -136
// CHECK-NEXT:      scfgwi t5, 256
// CHECK-NEXT:      li t5, -120
// CHECK-NEXT:      scfgwi t5, 288
// CHECK-NEXT:      li t5, 8
// CHECK-NEXT:      li a4, 2
// CHECK-NEXT:      li a3, 2
// CHECK-NEXT:      li t6, 35
// CHECK-NEXT:      scfgwi a4, 65
// CHECK-NEXT:      scfgwi a3, 97
// CHECK-NEXT:      scfgwi t6, 129
// CHECK-NEXT:      scfgwi t5, 193
// CHECK-NEXT:      li t5, 8
// CHECK-NEXT:      scfgwi t5, 225
// CHECK-NEXT:      li t5, -64
// CHECK-NEXT:      scfgwi t5, 257
// CHECK-NEXT:      scfgwi t4, 864
// CHECK-NEXT:      scfgwi t2, 833
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t2, 288
// CHECK-NEXT:      mv t1, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_0_for:
// CHECK-NEXT:      add t4, t0, t1
// CHECK-NEXT:      fld ft3, 0(t4)
// CHECK-NEXT:      li t5, 8
// CHECK-NEXT:      frep.o t5, 1, 0, 0
// CHECK-NEXT:      fmadd.d ft3, ft0, ft1, ft3
// CHECK-NEXT:      fsd ft3, 0(t4)
// CHECK-NEXT:      addi t1, t1, 8
// CHECK-NEXT:      blt t1, t2, scf_body_0_for
// CHECK-NEXT:  scf_body_end_0_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret


  func.func @ddot(
    %X : memref<128xf64>,
    %Y : memref<128xf64>,
    %G : memref<f64>
  ) {
    %G_moved = builtin.unrealized_conversion_cast %G : memref<f64> to !riscv.reg<>

    memref_stream.streaming_region {
      bounds = [128],
      indexing_maps = [
          affine_map<(m) -> (m)>,
          affine_map<(m) -> (m)>
      ]
    } ins(%X, %Y : memref<128xf64>, memref<128xf64>) {
    ^0(%x_stream : !stream.readable<f64>, %y_stream : !stream.readable<f64>):
      %X_stream = builtin.unrealized_conversion_cast %x_stream : !stream.readable<f64> to !stream.readable<!riscv.freg<>>
      %Y_stream = builtin.unrealized_conversion_cast %y_stream : !stream.readable<f64> to !stream.readable<!riscv.freg<>>
        %init = riscv.fld %G_moved, 0 : (!riscv.reg<>) -> !riscv.freg<>

        %c0 = riscv.li 0: () -> !riscv.reg<>
        %c1 = riscv.li 1: () -> !riscv.reg<>
        %c128 = riscv.li 128: () -> !riscv.reg<>
        %g = riscv_scf.for %i : !riscv.reg<> = %c0 to %c128 step %c1 iter_args(%acc = %init) -> (!riscv.freg<>) {
          %x = riscv_snitch.read from %X_stream : !riscv.freg<>
          %y = riscv_snitch.read from %Y_stream : !riscv.freg<>
          %res = riscv.fmadd.d %x, %y, %acc : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
          riscv_scf.yield %res : !riscv.freg<>
        }

        riscv.fsd %G_moved, %g, 0 : (!riscv.reg<>, !riscv.freg<>) -> ()
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
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      li t4, 127
// CHECK-NEXT:      scfgwi t4, 95
// CHECK-NEXT:      scfgwi t3, 223
// CHECK-NEXT:      scfgwi t2, 768
// CHECK-NEXT:      scfgwi t1, 769
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      fld ft3, 0(t0)
// CHECK-NEXT:      li t1, 127
// CHECK-NEXT:      frep.o t1, 1, 0, 0
// CHECK-NEXT:      fmadd.d ft3, ft0, ft1, ft3
// CHECK-NEXT:      fsd ft3, 0(t0)
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret


  // * Inputs:  x[ M x K ]
  // * Weights: w[ K x N ]
  // * Biases:  b[ M x N ]
  // * Outputs: y[ M x N ]
  func.func @dense(
    %X : memref<8x8xf64>,
    %W : memref<8x8xf64>,
    %B : memref<8x8xf64>,
    %Y : memref<8x8xf64>
  ) {
    %Y_moved = builtin.unrealized_conversion_cast %Y : memref<8x8xf64> to !riscv.reg<>

    %c0 = riscv.li 0 : () -> !riscv.reg<>
    %c1 = riscv.li 1 : () -> !riscv.reg<>
    %c8 = riscv.li 8 : () -> !riscv.reg<>
    %c512 = riscv.li 512 : () -> !riscv.reg<>

    %zero_float = riscv.fcvt.d.w %c0 : (!riscv.reg<>) -> !riscv.freg<>

    memref_stream.streaming_region {
      bounds = [8, 8, 8],
      indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n) -> (m, n)>
      ]
    } ins(%X, %W, %B : memref<8x8xf64>, memref<8x8xf64>, memref<8x8xf64>) {
    ^0(%x_stream : !stream.readable<f64>, %w_stream : !stream.readable<f64>, %b_stream : !stream.readable<f64>):
      %X_stream = builtin.unrealized_conversion_cast %x_stream : !stream.readable<f64> to !stream.readable<!riscv.freg<>>
      %W_stream = builtin.unrealized_conversion_cast %w_stream : !stream.readable<f64> to !stream.readable<!riscv.freg<>>
      %B_stream = builtin.unrealized_conversion_cast %b_stream : !stream.readable<f64> to !stream.readable<!riscv.freg<>>

      riscv_scf.for %y_i : !riscv.reg<> = %c0 to %c512 step %c8 {
        %Y_dest = riscv.add %Y_moved, %y_i : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %c = riscv.fld %Y_dest, 0 : (!riscv.reg<>) -> !riscv.freg<>

        %dot = riscv_scf.for %i : !riscv.reg<> = %c0 to %c8 step %c1 iter_args(%acc = %c) -> (!riscv.freg<>) {
          %x = riscv_snitch.read from %X_stream : !riscv.freg<>
          %w = riscv_snitch.read from %W_stream : !riscv.freg<>
          %res = riscv.fmadd.d %x, %w, %acc : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
          riscv_scf.yield %res : !riscv.freg<>
        }

        %b = riscv.get_float_register : () -> !riscv.freg<>
        %y_0 = riscv.fadd.d %b, %dot : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
        %y_1 = riscv.fmax.d %y_0, %zero_float : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>

        riscv.fsd %Y_dest, %y_1, 0 : (!riscv.reg<>, !riscv.freg<>) -> ()

        riscv_scf.yield
      }
    }

    func.return
  }


// CHECK:       .text
// CHECK-NEXT:  .globl dense
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  dense:
// CHECK-NEXT:      mv t6, a0
// CHECK-NEXT:      mv t5, a1
// CHECK-NEXT:      mv t4, a2
// CHECK-NEXT:      mv t0, a3
// CHECK-NEXT:      li t2, 512
// CHECK-NEXT:      fcvt.d.w ft3, zero
// CHECK-NEXT:      li a4, 8
// CHECK-NEXT:      li a5, 7
// CHECK-NEXT:      li a6, 7
// CHECK-NEXT:      li a7, 7
// CHECK-NEXT:      scfgwi a5, 64
// CHECK-NEXT:      scfgwi a6, 96
// CHECK-NEXT:      scfgwi a7, 128
// CHECK-NEXT:      scfgwi a4, 192
// CHECK-NEXT:      li a4, -56
// CHECK-NEXT:      scfgwi a4, 224
// CHECK-NEXT:      li a4, 8
// CHECK-NEXT:      scfgwi a4, 256
// CHECK-NEXT:      li a4, 64
// CHECK-NEXT:      li a7, 7
// CHECK-NEXT:      li a6, 7
// CHECK-NEXT:      li a5, 7
// CHECK-NEXT:      scfgwi a7, 65
// CHECK-NEXT:      scfgwi a6, 97
// CHECK-NEXT:      scfgwi a5, 129
// CHECK-NEXT:      scfgwi a4, 193
// CHECK-NEXT:      li a4, -440
// CHECK-NEXT:      scfgwi a4, 225
// CHECK-NEXT:      li a4, -504
// CHECK-NEXT:      scfgwi a4, 257
// CHECK-NEXT:      li a4, 8
// CHECK-NEXT:      li a5, 63
// CHECK-NEXT:      scfgwi a5, 66
// CHECK-NEXT:      scfgwi a4, 194
// CHECK-NEXT:      scfgwi t6, 832
// CHECK-NEXT:      scfgwi t5, 833
// CHECK-NEXT:      scfgwi t4, 770
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      mv t1, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_1_for:
// CHECK-NEXT:      add t4, t0, t1
// CHECK-NEXT:      fld ft5, 0(t4)
// CHECK-NEXT:      li t5, 7
// CHECK-NEXT:      frep.o t5, 1, 0, 0
// CHECK-NEXT:      fmadd.d ft5, ft0, ft1, ft5
// CHECK-NEXT:      fadd.d ft4, ft4, ft5
// CHECK-NEXT:      fmax.d ft4, ft4, ft3
// CHECK-NEXT:      fsd ft4, 0(t4)
// CHECK-NEXT:      addi t1, t1, 8
// CHECK-NEXT:      blt t1, t2, scf_body_1_for
// CHECK-NEXT:  scf_body_end_1_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret

    func.func @dsum(
      %X : memref<8x16xf64>,
      %Y : memref<8x16xf64>,
      %Z : memref<8x16xf64>
    ) {
      %c0 = riscv.li 0 : () -> !riscv.reg<>
      %c1 = riscv.li 1 : () -> !riscv.reg<>
      %c128 = riscv.li 128 : () -> !riscv.reg<>

      memref_stream.streaming_region {
        bounds = [16, 16],
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ]
      } ins(%X, %Y : memref<8x16xf64>, memref<8x16xf64>) outs(%Z : memref<8x16xf64>) {
      ^0(%x_stream : !stream.readable<f64>, %y_stream : !stream.readable<f64>, %z_stream : !stream.writable<f64>):
        %X_stream = builtin.unrealized_conversion_cast %x_stream : !stream.readable<f64> to !stream.readable<!riscv.freg<>>
        %Y_stream = builtin.unrealized_conversion_cast %y_stream : !stream.readable<f64> to !stream.readable<!riscv.freg<>>
        %Z_stream = builtin.unrealized_conversion_cast %z_stream : !stream.writable<f64> to !stream.writable<!riscv.freg<>>

        riscv_scf.for %i : !riscv.reg<> = %c0 to %c128 step %c1 {
          %9 = riscv_snitch.read from %X_stream : !riscv.freg<>
          %10 = riscv_snitch.read from %Y_stream : !riscv.freg<>
          %11 = riscv.fadd.d %9, %10 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
          riscv_snitch.write %11 to %Z_stream : !riscv.freg<>
          riscv_scf.yield
        }
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
// CHECK-NEXT:      li t3, 8
// CHECK-NEXT:      li t4, 255
// CHECK-NEXT:      scfgwi t4, 95
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
    %X_moved = builtin.unrealized_conversion_cast %X : f64 to !riscv.freg<>

    %x = riscv.fmv.d %X_moved : (!riscv.freg<>) -> !riscv.freg<>

    memref_stream.streaming_region {
      bounds = [16, 16],
      indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>
      ]
    } outs(%Y : memref<16x16xf64>) {
    ^0(%y_stream : !stream.writable<f64>):
      %Y_stream = builtin.unrealized_conversion_cast %y_stream : !stream.writable<f64> to !stream.writable<!riscv.freg<>>

      %c0 = riscv.li 0 : () -> !riscv.reg<>
      %c1 = riscv.li 1 : () -> !riscv.reg<>
      %c256 = riscv.li 256 : () -> !riscv.reg<>
      riscv_scf.for %i : !riscv.reg<> = %c0 to %c256 step %c1 {
        %y = riscv.fmv.d %x : (!riscv.freg<>) -> !riscv.freg<>
        riscv_snitch.write %y to %Y_stream : !riscv.freg<>
      }
    }

    func.return
  }


// CHECK:       .text
// CHECK-NEXT:  .globl fill
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  fill:
// CHECK-NEXT:      fmv.d ft3, fa0
// CHECK-NEXT:      mv t0, a0
// CHECK-NEXT:      li t1, 8
// CHECK-NEXT:      li t2, 255
// CHECK-NEXT:      scfgwi t2, 64
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
    %G_moved = builtin.unrealized_conversion_cast %G : memref<8x8xf64> to !riscv.reg<>

    %c0 = riscv.li 0 : () -> !riscv.reg<>
    %c1 = riscv.li 1 : () -> !riscv.reg<>
    %c8 = riscv.li 8 : () -> !riscv.reg<>
    %c512 = riscv.li 512 : () -> !riscv.reg<>

    memref_stream.streaming_region {
      bounds = [8, 8, 8],
      indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>
      ]
    } ins(%X, %Y : memref<8x8xf64>, memref<8x8xf64>) {
    ^0(%x_stream : !stream.readable<f64>, %y_stream : !stream.readable<f64>):
      %X_stream = builtin.unrealized_conversion_cast %x_stream : !stream.readable<f64> to !stream.readable<!riscv.freg<>>
      %Y_stream = builtin.unrealized_conversion_cast %y_stream : !stream.readable<f64> to !stream.readable<!riscv.freg<>>
      riscv_scf.for %g_i : !riscv.reg<> = %c0 to %c512 step %c8 {
        %G_dest = riscv.add %G_moved, %g_i : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %init = riscv.fld %G_dest, 0 : (!riscv.reg<>) -> !riscv.freg<>

        %g = riscv_scf.for %i : !riscv.reg<> = %c0 to %c8 step %c1 iter_args(%acc = %init) -> (!riscv.freg<>) {
          %x = riscv_snitch.read from %X_stream : !riscv.freg<>
          %y = riscv_snitch.read from %Y_stream : !riscv.freg<>
          %res = riscv.fmadd.d %x, %y, %acc : (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
          riscv_scf.yield %res : !riscv.freg<>
        }

        riscv.fsd %G_dest, %g, 0 : (!riscv.reg<>, !riscv.freg<>) -> ()

        riscv_scf.yield
      }
    }

    func.return
  }


// CHECK:       .text
// CHECK-NEXT:  .globl matmul
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  matmul:
// CHECK-NEXT:      mv t5, a0
// CHECK-NEXT:      mv t4, a1
// CHECK-NEXT:      mv t0, a2
// CHECK-NEXT:      li t2, 512
// CHECK-NEXT:      li t6, 8
// CHECK-NEXT:      li a3, 7
// CHECK-NEXT:      li a4, 7
// CHECK-NEXT:      li a5, 7
// CHECK-NEXT:      scfgwi a3, 64
// CHECK-NEXT:      scfgwi a4, 96
// CHECK-NEXT:      scfgwi a5, 128
// CHECK-NEXT:      scfgwi t6, 192
// CHECK-NEXT:      li t6, -56
// CHECK-NEXT:      scfgwi t6, 224
// CHECK-NEXT:      li t6, 8
// CHECK-NEXT:      scfgwi t6, 256
// CHECK-NEXT:      li t6, 64
// CHECK-NEXT:      li a5, 7
// CHECK-NEXT:      li a4, 7
// CHECK-NEXT:      li a3, 7
// CHECK-NEXT:      scfgwi a5, 65
// CHECK-NEXT:      scfgwi a4, 97
// CHECK-NEXT:      scfgwi a3, 129
// CHECK-NEXT:      scfgwi t6, 193
// CHECK-NEXT:      li t6, -440
// CHECK-NEXT:      scfgwi t6, 225
// CHECK-NEXT:      li t6, -504
// CHECK-NEXT:      scfgwi t6, 257
// CHECK-NEXT:      scfgwi t5, 832
// CHECK-NEXT:      scfgwi t4, 833
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      mv t1, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_{{\d+}}_for:
// CHECK-NEXT:      add t4, t0, t1
// CHECK-NEXT:      fld ft3, 0(t4)
// CHECK-NEXT:      li t5, 7
// CHECK-NEXT:      frep.o t5, 1, 0, 0
// CHECK-NEXT:      fmadd.d ft3, ft0, ft1, ft3
// CHECK-NEXT:      fsd ft3, 0(t4)
// CHECK-NEXT:      addi t1, t1, 8
// CHECK-NEXT:      blt t1, t2, scf_body_{{\d+}}_for
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
    %Y_moved = builtin.unrealized_conversion_cast %Y : memref<1x1x7x7xf64> to !riscv.reg<>

    %c0 = riscv.li 0 : () -> !riscv.reg<>
    %c1 = riscv.li 1 : () -> !riscv.reg<>
    %c8 = riscv.li 8 : () -> !riscv.reg<>
    %c9 = riscv.li 9 : () -> !riscv.reg<>
    %c512 = riscv.li 512 : () -> !riscv.reg<>

    memref_stream.streaming_region {
      bounds = [1, 1, 7, 7, 3, 3],
      indexing_maps = [
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d3 * 2 + d5)>
      ]
    } ins(%X : memref<1x1x16x16xf64>) {
    ^0(%x_stream : !stream.readable<f64>):
      %X_stream = builtin.unrealized_conversion_cast %x_stream : !stream.readable<f64> to !stream.readable<!riscv.freg<>>

      %c392 = riscv.li 392 : () -> !riscv.reg<>
      riscv_scf.for %y_i : !riscv.reg<> = %c0 to %c392 step %c8 {
        %Y_dest = riscv.add %Y_moved, %y_i : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %init = riscv.fld %Y_dest, 0 : (!riscv.reg<>) -> !riscv.freg<>

        %y = riscv_scf.for %i : !riscv.reg<> = %c0 to %c9 step %c1 iter_args(%acc = %init) -> (!riscv.freg<>) {
          %x = riscv_snitch.read from %X_stream : !riscv.freg<>
          %res = riscv.fmax.d %x, %acc : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
          riscv_scf.yield %res : !riscv.freg<>
        }

        riscv.fsd %Y_dest, %y, 0 : (!riscv.reg<>, !riscv.freg<>) -> ()

        riscv_scf.yield
      }
    }

    func.return
  }


// CHECK:       .text
// CHECK-NEXT:  .globl pooling_nchw_max_d1_s2_3x3
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  pooling_nchw_max_d1_s2_3x3:
// CHECK-NEXT:      mv t2, a0
// CHECK-NEXT:      mv t0, a1
// CHECK-NEXT:      li t4, 8
// CHECK-NEXT:      li a3, 2
// CHECK-NEXT:      li a2, 2
// CHECK-NEXT:      li t6, 6
// CHECK-NEXT:      li t5, 6
// CHECK-NEXT:      scfgwi a3, 64
// CHECK-NEXT:      scfgwi a2, 96
// CHECK-NEXT:      scfgwi t6, 128
// CHECK-NEXT:      scfgwi t5, 160
// CHECK-NEXT:      scfgwi t4, 192
// CHECK-NEXT:      li t4, 112
// CHECK-NEXT:      scfgwi t4, 224
// CHECK-NEXT:      li t4, -256
// CHECK-NEXT:      scfgwi t4, 256
// CHECK-NEXT:      li t4, -112
// CHECK-NEXT:      scfgwi t4, 288
// CHECK-NEXT:      scfgwi t2, 864
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t2, 392
// CHECK-NEXT:      mv t1, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_{{\d+}}_for:
// CHECK-NEXT:      add t4, t0, t1
// CHECK-NEXT:      fld ft3, 0(t4)
// CHECK-NEXT:      li t5, 8
// CHECK-NEXT:      frep.o t5, 1, 0, 0
// CHECK-NEXT:      fmax.d ft3, ft0, ft3
// CHECK-NEXT:      fsd ft3, 0(t4)
// CHECK-NEXT:      addi t1, t1, 8
// CHECK-NEXT:      blt t1, t2, scf_body_{{\d+}}_for
// CHECK-NEXT:  scf_body_end_{{\d+}}_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret


  func.func public @relu(%X: memref<16x16xf64>, %Y: memref<16x16xf64>) {

    %zero_int = riscv.get_register : () -> !riscv.reg<zero>
    %zero_float = riscv.fcvt.d.w %zero_int : (!riscv.reg<zero>) -> !riscv.freg<>

    memref_stream.streaming_region {
      bounds = [16, 16],
      indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
      ]
    } ins(%X : memref<16x16xf64>) outs(%Y : memref<16x16xf64>) {
    ^0(%x_stream : !stream.readable<f64>, %y_stream : !stream.writable<f64>):
      %X_stream = builtin.unrealized_conversion_cast %x_stream : !stream.readable<f64> to !stream.readable<!riscv.freg<>>
      %Y_stream = builtin.unrealized_conversion_cast %y_stream : !stream.writable<f64> to !stream.writable<!riscv.freg<>>
      %c0 = riscv.li 0 : () -> !riscv.reg<>
      %c1 = riscv.li 1 : () -> !riscv.reg<>
      %c256 = riscv.li 256 : () -> !riscv.reg<>
      riscv_scf.for %i : !riscv.reg<> = %c0 to %c256 step %c1 {
        %x = riscv_snitch.read from %X_stream : !riscv.freg<>
        %y = riscv.fmax.d %x, %zero_float : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
        riscv_snitch.write %y to %Y_stream : !riscv.freg<>
      }
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
// CHECK-NEXT:      li t2, 8
// CHECK-NEXT:      li t3, 255
// CHECK-NEXT:      scfgwi t3, 95
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
    %X_moved = builtin.unrealized_conversion_cast %X : memref<1x1x16x16xf64> to !riscv.reg<>
    %Y_moved = builtin.unrealized_conversion_cast %Y : memref<1x1x7x7xf64> to !riscv.reg<>

    %c0 = riscv.li 0 : () -> !riscv.reg<>
    %c1 = riscv.li 1 : () -> !riscv.reg<>
    %c8 = riscv.li 8 : () -> !riscv.reg<>
    %c9 = riscv.li 9 : () -> !riscv.reg<>
    %c512 = riscv.li 512 : () -> !riscv.reg<>

    memref_stream.streaming_region {
      bounds = [1, 1, 7, 7, 3, 3],
      indexing_maps = [
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d3 * 2 + d5)>
      ]
    } ins(%X : memref<1x1x16x16xf64>) {
    ^0(%x_stream : !stream.readable<f64>):
      %X_stream = builtin.unrealized_conversion_cast %x_stream : !stream.readable<f64> to !stream.readable<!riscv.freg<>>

      %c392 = riscv.li 392 : () -> !riscv.reg<>
      riscv_scf.for %y_i : !riscv.reg<> = %c0 to %c392 step %c8 {
        %Y_dest = riscv.add %Y_moved, %y_i : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %init = riscv.fld %Y_dest, 0 : (!riscv.reg<>) -> !riscv.freg<>

        %y = riscv_scf.for %i : !riscv.reg<> = %c0 to %c9 step %c1 iter_args(%acc = %init) -> (!riscv.freg<>) {
          %x = riscv_snitch.read from %X_stream : !riscv.freg<>
          %res = riscv.fadd.d %x, %acc : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
          riscv_scf.yield %res : !riscv.freg<>
        }

        riscv.fsd %Y_dest, %y, 0 : (!riscv.reg<>, !riscv.freg<>) -> ()

        riscv_scf.yield
      }

    }

    func.return
  }


// CHECK:       .text
// CHECK-NEXT:  .globl pooling_nchw_sum_d1_s2_3x3
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  pooling_nchw_sum_d1_s2_3x3:
// CHECK-NEXT:      mv t2, a0
// CHECK-NEXT:      mv t0, a1
// CHECK-NEXT:      li t4, 8
// CHECK-NEXT:      li a3, 2
// CHECK-NEXT:      li a2, 2
// CHECK-NEXT:      li t6, 6
// CHECK-NEXT:      li t5, 6
// CHECK-NEXT:      scfgwi a3, 64
// CHECK-NEXT:      scfgwi a2, 96
// CHECK-NEXT:      scfgwi t6, 128
// CHECK-NEXT:      scfgwi t5, 160
// CHECK-NEXT:      scfgwi t4, 192
// CHECK-NEXT:      li t4, 112
// CHECK-NEXT:      scfgwi t4, 224
// CHECK-NEXT:      li t4, -256
// CHECK-NEXT:      scfgwi t4, 256
// CHECK-NEXT:      li t4, -112
// CHECK-NEXT:      scfgwi t4, 288
// CHECK-NEXT:      scfgwi t2, 864
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      li t2, 392
// CHECK-NEXT:      mv t1, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_{{\d+}}_for:
// CHECK-NEXT:      add t4, t0, t1
// CHECK-NEXT:      fld ft3, 0(t4)
// CHECK-NEXT:      li t5, 8
// CHECK-NEXT:      frep.o t5, 1, 0, 0
// CHECK-NEXT:      fadd.d ft3, ft0, ft3
// CHECK-NEXT:      fsd ft3, 0(t4)
// CHECK-NEXT:      addi t1, t1, 8
// CHECK-NEXT:      blt t1, t2, scf_body_{{\d+}}_for
// CHECK-NEXT:  scf_body_end_{{\d+}}_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret
