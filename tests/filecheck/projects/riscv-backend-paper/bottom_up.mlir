// RUN: xdsl-opt -p convert-arith-to-riscv,convert-func-to-riscv-func,convert-memref-stream-to-snitch,reconcile-unrealized-casts,test-lower-snitch-stream-to-asm -t riscv-asm %s | filecheck %s

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

    %c0_val = arith.constant 0 : i32
    %c0 = builtin.unrealized_conversion_cast %c0_val : i32 to !riscv.reg<>
    %c1_val = arith.constant 1 : i32
    %c1 = builtin.unrealized_conversion_cast %c1_val : i32 to !riscv.reg<>
    %c8_val = arith.constant 8 : i32
    %c8 = builtin.unrealized_conversion_cast %c8_val : i32 to !riscv.reg<>
    %c9_val = arith.constant 9 : i32
    %c9 = builtin.unrealized_conversion_cast %c9_val : i32 to !riscv.reg<>

    memref_stream.streaming_region {
      patterns = [
          #memref_stream.stride_pattern<ub = [1, 1, 6, 6, 1, 3, 3], index_map = (d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>,
          #memref_stream.stride_pattern<ub = [1, 1, 6, 6, 1, 3, 3], index_map = (d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
      ]
    } ins(%X, %Y : memref<1x1x8x8xf64>, memref<1x1x3x3xf64>) {
    ^0(%x_stream : !stream.readable<f64>, %y_stream : !stream.readable<f64>):

      %c288_val = arith.constant 288 : i32
      %c288 = builtin.unrealized_conversion_cast %c288_val : i32 to !riscv.reg<>
      riscv_scf.for %z_i : !riscv.reg<> = %c0 to %c288 step %c8 {
        %Z_dest = riscv.add %Z_moved, %z_i : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %c = riscv.fld %Z_dest, 0 : (!riscv.reg<>) -> !riscv.freg<>

        %z = riscv_scf.for %i : !riscv.reg<> = %c0 to %c9 step %c1 iter_args(%acc = %c) -> (!riscv.freg<>) {
          %x = memref_stream.read from %x_stream : f64
          %y = memref_stream.read from %y_stream : f64
          %acc_val = builtin.unrealized_conversion_cast %acc : !riscv.freg<> to f64
          %prod = arith.mulf %x, %y fastmath<fast> : f64
          %res_val = arith.addf %prod, %acc_val fastmath<fast> : f64
          %res = builtin.unrealized_conversion_cast %res_val : f64 to !riscv.freg<>
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
// CHECK-NEXT:  scf_body_{{\d+}}_for:
// CHECK-NEXT:      add t4, t0, t1
// CHECK-NEXT:      fld ft3, 0(t4)
// CHECK-NEXT:      li t5, 8
// CHECK-NEXT:      frep.o t5, 1, 0, 0
// CHECK-NEXT:      fmadd.d ft3, ft0, ft1, ft3
// CHECK-NEXT:      fsd ft3, 0(t4)
// CHECK-NEXT:      addi t1, t1, 8
// CHECK-NEXT:      blt t1, t2, scf_body_{{\d+}}_for
// CHECK-NEXT:  scf_body_end_{{\d+}}_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret


  func.func @ddot(
    %X : memref<128xf64>,
    %Y : memref<128xf64>,
    %G : memref<f64>
  ) {
    %G_moved = builtin.unrealized_conversion_cast %G : memref<f64> to !riscv.reg<>

    memref_stream.streaming_region {
      patterns = [
          #memref_stream.stride_pattern<ub = [128], index_map = (d0) -> (d0)>,
          #memref_stream.stride_pattern<ub = [128], index_map = (d0) -> (d0)>
      ]
    } ins(%X, %Y : memref<128xf64>, memref<128xf64>) {
    ^0(%x_stream : !stream.readable<f64>, %y_stream : !stream.readable<f64>):
        %init = riscv.fld %G_moved, 0 : (!riscv.reg<>) -> !riscv.freg<>

        %c0 = riscv.li 0: () -> !riscv.reg<>
        %c1 = riscv.li 1: () -> !riscv.reg<>
        %c128 = riscv.li 128: () -> !riscv.reg<>
        %g = riscv_scf.for %i : !riscv.reg<> = %c0 to %c128 step %c1 iter_args(%acc = %init) -> (!riscv.freg<>) {
          %x = memref_stream.read from %x_stream : f64
          %y = memref_stream.read from %y_stream : f64
          %acc_val = builtin.unrealized_conversion_cast %acc : !riscv.freg<> to f64
          %prod = arith.mulf %x, %y fastmath<fast> : f64
          %res_val = arith.addf %prod, %acc_val fastmath<fast> : f64
          %res = builtin.unrealized_conversion_cast %res_val : f64 to !riscv.freg<>
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

    %c0_val = arith.constant 0 : i32
    %c0 = builtin.unrealized_conversion_cast %c0_val : i32 to !riscv.reg<>
    %c1_val = arith.constant 1 : i32
    %c1 = builtin.unrealized_conversion_cast %c1_val : i32 to !riscv.reg<>
    %c8_val = arith.constant 8 : i32
    %c8 = builtin.unrealized_conversion_cast %c8_val : i32 to !riscv.reg<>
    %c512_val = arith.constant 512 : i32
    %c512 = builtin.unrealized_conversion_cast %c512_val : i32 to !riscv.reg<>

    %zero_int = arith.constant 0 : i32
    %zero_float = arith.sitofp %zero_int : i32 to f64

    memref_stream.streaming_region {
      patterns = [
        #memref_stream.stride_pattern<ub = [8, 8, 8], index_map = (m, n, k) -> (m, k)>,
        #memref_stream.stride_pattern<ub = [8, 8, 8], index_map = (m, n, k) -> (k, n)>,
        #memref_stream.stride_pattern<ub = [8, 8], index_map = (m, n) -> (m, n)>
      ]
    } ins(%X, %W, %B : memref<8x8xf64>, memref<8x8xf64>, memref<8x8xf64>) {
    ^0(%x_stream : !stream.readable<f64>, %w_stream : !stream.readable<f64>, %b_stream : !stream.readable<f64>):

      riscv_scf.for %y_i : !riscv.reg<> = %c0 to %c512 step %c8 {
        %Y_dest = riscv.add %Y_moved, %y_i : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %c = riscv.fld %Y_dest, 0 : (!riscv.reg<>) -> !riscv.freg<>

        %dot = riscv_scf.for %i : !riscv.reg<> = %c0 to %c8 step %c1 iter_args(%acc = %c) -> (!riscv.freg<>) {
          %x = memref_stream.read from %x_stream : f64
          %w = memref_stream.read from %w_stream : f64
          %acc_val = builtin.unrealized_conversion_cast %acc : !riscv.freg<> to f64
          %prod = arith.mulf %x, %w fastmath<fast> : f64
          %res_val = arith.addf %prod, %acc_val fastmath<fast> : f64
          %res = builtin.unrealized_conversion_cast %res_val : f64 to !riscv.freg<>
          riscv_scf.yield %res : !riscv.freg<>
        }

        %b_val = memref_stream.read from %b_stream : f64
        %dot_val = builtin.unrealized_conversion_cast %dot : !riscv.freg<> to f64
        %y_0_val = arith.addf %b_val, %dot_val : f64
        %y_1_val = arith.maximumf %y_0_val, %zero_float : f64
        %y_1 = builtin.unrealized_conversion_cast %y_1_val : f64 to !riscv.freg<>

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
// CHECK-NEXT:  scf_body_{{\d+}}_for:
// CHECK-NEXT:      add t4, t0, t1
// CHECK-NEXT:      fld ft4, 0(t4)
// CHECK-NEXT:      li t5, 7
// CHECK-NEXT:      frep.o t5, 1, 0, 0
// CHECK-NEXT:      fmadd.d ft4, ft0, ft1, ft4
// CHECK-NEXT:      fadd.d ft4, ft2, ft4
// CHECK-NEXT:      fmax.d ft4, ft4, ft3
// CHECK-NEXT:      fsd ft4, 0(t4)
// CHECK-NEXT:      addi t1, t1, 8
// CHECK-NEXT:      blt t1, t2, scf_body_{{\d+}}_for
// CHECK-NEXT:  scf_body_end_{{\d+}}_for:
// CHECK-NEXT:      csrrci zero, 1984, 1
// CHECK-NEXT:      ret

    func.func @dsum(
      %X : memref<8x16xf64>,
      %Y : memref<8x16xf64>,
      %Z : memref<8x16xf64>
    ) {
      %c0_val = arith.constant 0 : i32
      %c0 = builtin.unrealized_conversion_cast %c0_val : i32 to !riscv.reg<>
      %c1_val = arith.constant 1 : i32
      %c1 = builtin.unrealized_conversion_cast %c1_val : i32 to !riscv.reg<>
      %c128_val = arith.constant 128 : i32
      %c128 = builtin.unrealized_conversion_cast %c128_val : i32 to !riscv.reg<>

      memref_stream.streaming_region {
        patterns = [
          #memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>,
          #memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>,
          #memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>
        ]
      } ins(%X, %Y : memref<8x16xf64>, memref<8x16xf64>) outs(%Z : memref<8x16xf64>) {
      ^0(%x_stream : !stream.readable<f64>, %y_stream : !stream.readable<f64>, %z_stream : !stream.writable<f64>):

        riscv_scf.for %i : !riscv.reg<> = %c0 to %c128 step %c1 {
          %x = memref_stream.read from %x_stream : f64
          %y = memref_stream.read from %y_stream : f64
          %z = arith.addf %x, %y : f64
          memref_stream.write %z to %z_stream : f64
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
// CHECK-NEXT:      li t4, 127
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
      patterns = [
        #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>
      ]
    } outs(%Y : memref<16x16xf64>) {
    ^0(%y_stream : !stream.writable<f64>):

      %c0_val = arith.constant 0 : i32
      %c0 = builtin.unrealized_conversion_cast %c0_val : i32 to !riscv.reg<>
      %c1_val = arith.constant 1 : i32
      %c1 = builtin.unrealized_conversion_cast %c1_val : i32 to !riscv.reg<>
      %c256_val = arith.constant 256 : i32
      %c256 = builtin.unrealized_conversion_cast %c256_val : i32 to !riscv.reg<>
      riscv_scf.for %i : !riscv.reg<> = %c0 to %c256 step %c1 {
        %y = riscv.fmv.d %x : (!riscv.freg<>) -> !riscv.freg<>
        %y_val = builtin.unrealized_conversion_cast %y : !riscv.freg<> to f64
        memref_stream.write %y_val to %y_stream : f64
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
    %c0 = riscv.get_register : () -> !riscv.reg<zero>
    %c1_val = arith.constant 1 : i32
    %c1 = builtin.unrealized_conversion_cast %c1_val : i32 to !riscv.reg<>
    %c4_val = arith.constant 4 : i32
    %c4 = builtin.unrealized_conversion_cast %c4_val : i32 to !riscv.reg<>
    %frep_count = riscv.li 6 : () -> !riscv.reg<>
    %target_count = riscv.li 64 : () -> !riscv.reg<>

    memref_stream.streaming_region {
      patterns = [
        #memref_stream.stride_pattern<ub = [8, 2, 8, 4], index_map = (m, n, k, j) -> (m, k)>,
        #memref_stream.stride_pattern<ub = [8, 2, 8, 4], index_map = (m, n, k, j) -> (k, n * 4 + j)>,
        #memref_stream.stride_pattern<ub = [8, 2, 4], index_map = (m, n, j) -> (m, n * 4 + j)>
      ]
    } ins(%X, %Y : memref<8x8xf64>, memref<8x8xf64>) outs(%G : memref<8x8xf64>) {
    ^0(%x_stream : !stream.readable<f64>, %y_stream : !stream.readable<f64>, %g_stream : !stream.writable<f64>):
      riscv_scf.for %g_i : !riscv.reg<> = %c0 to %target_count step %c4 {
        %x00 = memref_stream.read from %x_stream : f64
        %y00 = memref_stream.read from %y_stream : f64
        %init0_val = arith.mulf %x00, %y00 fastmath<fast> : f64
        %init0 = builtin.unrealized_conversion_cast %init0_val : f64 to !riscv.freg<>
        %x01 = memref_stream.read from %x_stream : f64
        %y01 = memref_stream.read from %y_stream : f64
        %init1_val = arith.mulf %x01, %y01 fastmath<fast> : f64
        %init1 = builtin.unrealized_conversion_cast %init1_val : f64 to !riscv.freg<>
        %x02 = memref_stream.read from %x_stream : f64
        %y02 = memref_stream.read from %y_stream : f64
        %init2_val = arith.mulf %x02, %y02 fastmath<fast> : f64
        %init2 = builtin.unrealized_conversion_cast %init2_val : f64 to !riscv.freg<>
        %x03 = memref_stream.read from %x_stream : f64
        %y03 = memref_stream.read from %y_stream : f64
        %init3_val = arith.mulf %x03, %y03 fastmath<fast> : f64
        %init3 = builtin.unrealized_conversion_cast %init3_val : f64 to !riscv.freg<>

        %g00, %g01, %g02, %g03 = riscv_scf.for %inner_i : !riscv.reg<> = %c0 to %frep_count step %c1 iter_args(%acc0 = %init0, %acc1 = %init1, %acc2 = %init2, %acc3 = %init3) -> (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>, !riscv.freg<>) {
          %x10 = memref_stream.read from %x_stream : f64
          %y10 = memref_stream.read from %y_stream : f64
          %acc0_val = builtin.unrealized_conversion_cast %acc0 : !riscv.freg<> to f64
          %prod10 = arith.mulf %x10, %y10 fastmath<fast> : f64
          %res0_val = arith.addf %prod10, %acc0_val fastmath<fast> : f64
          %res0 = builtin.unrealized_conversion_cast %res0_val : f64 to !riscv.freg<>
          %x11 = memref_stream.read from %x_stream : f64
          %y11 = memref_stream.read from %y_stream : f64
          %acc1_val = builtin.unrealized_conversion_cast %acc1 : !riscv.freg<> to f64
          %prod11 = arith.mulf %x11, %y11 fastmath<fast> : f64
          %res1_val = arith.addf %prod11, %acc1_val fastmath<fast> : f64
          %res1 = builtin.unrealized_conversion_cast %res1_val : f64 to !riscv.freg<>
          %x12 = memref_stream.read from %x_stream : f64
          %y12 = memref_stream.read from %y_stream : f64
          %acc2_val = builtin.unrealized_conversion_cast %acc2 : !riscv.freg<> to f64
          %prod12 = arith.mulf %x12, %y12 fastmath<fast> : f64
          %res2_val = arith.addf %prod12, %acc2_val fastmath<fast> : f64
          %res2 = builtin.unrealized_conversion_cast %res2_val : f64 to !riscv.freg<>
          %x13 = memref_stream.read from %x_stream : f64
          %y13 = memref_stream.read from %y_stream : f64
          %acc3_val = builtin.unrealized_conversion_cast %acc3 : !riscv.freg<> to f64
          %prod13 = arith.mulf %x13, %y13 fastmath<fast> : f64
          %res3_val = arith.addf %prod13, %acc3_val fastmath<fast> : f64
          %res3 = builtin.unrealized_conversion_cast %res3_val : f64 to !riscv.freg<>

          riscv_scf.yield %res0, %res1, %res2, %res3 : !riscv.freg<>, !riscv.freg<>, !riscv.freg<>, !riscv.freg<>
        }

        %x20 = memref_stream.read from %x_stream : f64
        %y20 = memref_stream.read from %y_stream : f64
        %g00_val = builtin.unrealized_conversion_cast %g00 : !riscv.freg<> to f64
        %prod20 = arith.mulf %x20, %y20 fastmath<fast> : f64
        %g10 = arith.addf %prod20, %g00_val fastmath<fast> : f64
        memref_stream.write %g10 to %g_stream : f64
        %x21 = memref_stream.read from %x_stream : f64
        %y21 = memref_stream.read from %y_stream : f64
        %g01_val = builtin.unrealized_conversion_cast %g01 : !riscv.freg<> to f64
        %prod21 = arith.mulf %x21, %y21 fastmath<fast> : f64
        %g11 = arith.addf %prod21, %g01_val fastmath<fast> : f64
        memref_stream.write %g11 to %g_stream : f64
        %x22 = memref_stream.read from %x_stream : f64
        %y22 = memref_stream.read from %y_stream : f64
        %g02_val = builtin.unrealized_conversion_cast %g02 : !riscv.freg<> to f64
        %prod22 = arith.mulf %x22, %y22 fastmath<fast> : f64
        %g12 = arith.addf %prod22, %g02_val fastmath<fast> : f64
        memref_stream.write %g12 to %g_stream : f64
        %x23 = memref_stream.read from %x_stream : f64
        %y23 = memref_stream.read from %y_stream : f64
        %g03_val = builtin.unrealized_conversion_cast %g03 : !riscv.freg<> to f64
        %prod23 = arith.mulf %x23, %y23 fastmath<fast> : f64
        %g13 = arith.addf %prod23, %g03_val fastmath<fast> : f64
        memref_stream.write %g13 to %g_stream : f64

        riscv_scf.yield
      }
    }

    func.return
}

// CHECK-NEXT:  .text
// CHECK-NEXT:  .globl matmul
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  matmul:
// CHECK-NEXT:      mv t5, a0
// CHECK-NEXT:      mv t4, a1
// CHECK-NEXT:      mv t3, a2
// CHECK-NEXT:      li t1, 64
// CHECK-NEXT:      li a4, 3
// CHECK-NEXT:      li a5, 7
// CHECK-NEXT:      li a6, 1
// CHECK-NEXT:      li t6, 7
// CHECK-NEXT:      scfgwi a4, 64
// CHECK-NEXT:      scfgwi a5, 96
// CHECK-NEXT:      scfgwi a6, 128
// CHECK-NEXT:      scfgwi t6, 160
// CHECK-NEXT:      scfgwi zero, 192
// CHECK-NEXT:      li t6, 8
// CHECK-NEXT:      scfgwi t6, 224
// CHECK-NEXT:      li t6, -56
// CHECK-NEXT:      scfgwi t6, 256
// CHECK-NEXT:      li t6, 8
// CHECK-NEXT:      scfgwi t6, 288
// CHECK-NEXT:      li t6, 8
// CHECK-NEXT:      li a6, 3
// CHECK-NEXT:      li a5, 7
// CHECK-NEXT:      li a4, 1
// CHECK-NEXT:      li a3, 7
// CHECK-NEXT:      scfgwi a6, 65
// CHECK-NEXT:      scfgwi a5, 97
// CHECK-NEXT:      scfgwi a4, 129
// CHECK-NEXT:      scfgwi a3, 161
// CHECK-NEXT:      scfgwi t6, 193
// CHECK-NEXT:      li t6, 40
// CHECK-NEXT:      scfgwi t6, 225
// CHECK-NEXT:      li t6, -440
// CHECK-NEXT:      scfgwi t6, 257
// CHECK-NEXT:      li t6, -504
// CHECK-NEXT:      scfgwi t6, 289
// CHECK-NEXT:      li t6, 8
// CHECK-NEXT:      li a3, 63
// CHECK-NEXT:      scfgwi a3, 66
// CHECK-NEXT:      scfgwi t6, 194
// CHECK-NEXT:      scfgwi t5, 864
// CHECK-NEXT:      scfgwi t4, 865
// CHECK-NEXT:      scfgwi t3, 898
// CHECK-NEXT:      csrrsi zero, 1984, 1
// CHECK-NEXT:      mv t0, zero
// CHECK-NEXT:      # Constant folded riscv_cf.bge
// CHECK-NEXT:  scf_body_2_for:
// CHECK-NEXT:      fmul.d ft6, ft0, ft1
// CHECK-NEXT:      fmul.d ft5, ft0, ft1
// CHECK-NEXT:      fmul.d ft4, ft0, ft1
// CHECK-NEXT:      fmul.d ft3, ft0, ft1
// CHECK-NEXT:      li t3, 5
// CHECK-NEXT:      frep.o t3, 4, 0, 0
// CHECK-NEXT:      fmadd.d ft6, ft0, ft1, ft6
// CHECK-NEXT:      fmadd.d ft5, ft0, ft1, ft5
// CHECK-NEXT:      fmadd.d ft4, ft0, ft1, ft4
// CHECK-NEXT:      fmadd.d ft3, ft0, ft1, ft3
// CHECK-NEXT:      fmadd.d ft2, ft0, ft1, ft6
// CHECK-NEXT:      fmadd.d ft2, ft0, ft1, ft5
// CHECK-NEXT:      fmadd.d ft2, ft0, ft1, ft4
// CHECK-NEXT:      fmadd.d ft2, ft0, ft1, ft3
// CHECK-NEXT:      addi t0, t0, 4
// CHECK-NEXT:      blt t0, t1, scf_body_2_for
// CHECK-NEXT:  scf_body_end_2_for:
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

    %c0_val = arith.constant 0 : i32
    %c0 = builtin.unrealized_conversion_cast %c0_val : i32 to !riscv.reg<>
    %c1_val = arith.constant 1 : i32
    %c1 = builtin.unrealized_conversion_cast %c1_val : i32 to !riscv.reg<>
    %c8_val = arith.constant 8 : i32
    %c8 = builtin.unrealized_conversion_cast %c8_val : i32 to !riscv.reg<>
    %c9_val = arith.constant 9 : i32
    %c9 = builtin.unrealized_conversion_cast %c9_val : i32 to !riscv.reg<>
    %c512_val = arith.constant 512 : i32
    %c512 = builtin.unrealized_conversion_cast %c512_val : i32 to !riscv.reg<>

    memref_stream.streaming_region {
      patterns = [
        #memref_stream.stride_pattern<ub = [1, 1, 7, 7, 3, 3], index_map = (d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d3 * 2 + d5)>
      ]
    } ins(%X : memref<1x1x16x16xf64>) {
    ^0(%x_stream : !stream.readable<f64>):

      %c392_val = arith.constant 392 : i32
      %c392 = builtin.unrealized_conversion_cast %c392_val : i32 to !riscv.reg<>
      riscv_scf.for %y_i : !riscv.reg<> = %c0 to %c392 step %c8 {
        %Y_dest = riscv.add %Y_moved, %y_i : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %init = riscv.fld %Y_dest, 0 : (!riscv.reg<>) -> !riscv.freg<>

        %y = riscv_scf.for %i : !riscv.reg<> = %c0 to %c9 step %c1 iter_args(%acc = %init) -> (!riscv.freg<>) {
          %x_val = memref_stream.read from %x_stream : f64
          %x = builtin.unrealized_conversion_cast %x_val : f64 to !riscv.freg<>
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

    %zero_int = arith.constant 0 : i32
    %zero_float = arith.sitofp %zero_int : i32 to f64

    memref_stream.streaming_region {
      patterns = [
          #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>,
          #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>
      ]
    } ins(%X : memref<16x16xf64>) outs(%Y : memref<16x16xf64>) {
    ^0(%x_stream : !stream.readable<f64>, %y_stream : !stream.writable<f64>):
      %c0_val = arith.constant 0 : i32
      %c0 = builtin.unrealized_conversion_cast %c0_val : i32 to !riscv.reg<>
      %c1_val = arith.constant 1 : i32
      %c1 = builtin.unrealized_conversion_cast %c1_val : i32 to !riscv.reg<>
      %c256_val = arith.constant 256 : i32
      %c256 = builtin.unrealized_conversion_cast %c256_val : i32 to !riscv.reg<>
      riscv_scf.for %i : !riscv.reg<> = %c0 to %c256 step %c1 {
        %x_val = memref_stream.read from %x_stream : f64
        %x = builtin.unrealized_conversion_cast %x_val : f64 to !riscv.freg<>
        %zero_reg = builtin.unrealized_conversion_cast %zero_float : f64 to !riscv.freg<>
        %y = riscv.fmax.d %x, %zero_reg : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
        %y_val = builtin.unrealized_conversion_cast %y : !riscv.freg<> to f64
        memref_stream.write %y_val to %y_stream : f64
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
    %Y_moved = builtin.unrealized_conversion_cast %Y : memref<1x1x7x7xf64> to !riscv.reg<>

    %c0_val = arith.constant 0 : i32
    %c0 = builtin.unrealized_conversion_cast %c0_val : i32 to !riscv.reg<>
    %c1_val = arith.constant 1 : i32
    %c1 = builtin.unrealized_conversion_cast %c1_val : i32 to !riscv.reg<>
    %c8_val = arith.constant 8 : i32
    %c8 = builtin.unrealized_conversion_cast %c8_val : i32 to !riscv.reg<>
    %c9_val = arith.constant 9 : i32
    %c9 = builtin.unrealized_conversion_cast %c9_val : i32 to !riscv.reg<>
    %c512_val = arith.constant 512 : i32
    %c512 = builtin.unrealized_conversion_cast %c512_val : i32 to !riscv.reg<>

    memref_stream.streaming_region {
      patterns = [
        #memref_stream.stride_pattern<ub = [1, 1, 7, 7, 3, 3], index_map = (d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d3 * 2 + d5)>
      ]
    } ins(%X : memref<1x1x16x16xf64>) {
    ^0(%x_stream : !stream.readable<f64>):

      %c392_val = arith.constant 392 : i32
      %c392 = builtin.unrealized_conversion_cast %c392_val : i32 to !riscv.reg<>
      riscv_scf.for %y_i : !riscv.reg<> = %c0 to %c392 step %c8 {
        %Y_dest = riscv.add %Y_moved, %y_i : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %init = riscv.fld %Y_dest, 0 : (!riscv.reg<>) -> !riscv.freg<>

        %y = riscv_scf.for %i : !riscv.reg<> = %c0 to %c9 step %c1 iter_args(%acc = %init) -> (!riscv.freg<>) {
          %x_val = memref_stream.read from %x_stream : f64
          %acc_val = builtin.unrealized_conversion_cast %acc : !riscv.freg<> to f64
          %res_val = arith.addf %x_val, %acc_val : f64
          %res = builtin.unrealized_conversion_cast %res_val : f64 to !riscv.freg<>
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
