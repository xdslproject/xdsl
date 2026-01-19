// RUN: xdsl-opt -p test-lower-linalg-to-snitch -t riscv-asm %s | filecheck %s

func.func public @ssum(
  %X: memref<8x16xf32>,
  %Y: memref<8x16xf32>,
  %Z: memref<8x16xf32>
) {
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%X, %Y : memref<8x16xf32>, memref<8x16xf32>) outs(%Z : memref<8x16xf32>) {
  ^bb1(%in : f32, %in_1 : f32, %out : f32):
    %3 = arith.addf %in, %in_1 : f32
    linalg.yield %3 : f32
  }
  func.return
}

// CHECK:           # Regalloc stats: {"preallocated_float": ["ft0", "ft1", "ft2"], "preallocated_int": ["a0", "a1", "a2", "zero"], "allocated_float": ["ft0", "ft1", "ft2", "ft3"], "allocated_int": ["a0", "a1", "a2", "t0", "t1", "t2", "t3", "zero"]}
// CHECK-NEXT:  .text
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
// CHECK-NEXT:      frep.o t0, 2, 0, 0
// CHECK-NEXT:      vfadd.s ft3, ft0, ft1
// CHECK-NEXT:      fmv.d ft2, ft3
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
    ^bb0(%x : f64, %alloc_val: f64, %acc : f64):
      %res = arith.maximumf %x, %acc : f64
      linalg.yield %res : f64
    }
    memref.dealloc %alloc : memref<3x3xf32>
    func.return
  }

// CHECK-NEXT:    # Regalloc stats: {"preallocated_float": ["ft0", "ft1", "ft2"], "preallocated_int": ["a0", "a1", "zero"], "allocated_float": ["ft0", "ft1", "ft3", "ft4", "ft5", "ft6", "ft7"], "allocated_int": ["a0", "a1", "a2", "a3", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "zero"]}
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


  riscv.assembly_section ".text" {
    riscv_func.func public @reluf32(%X : !riscv.reg<a0>, %Y : !riscv.reg<a1>) attributes {p2align = 2 : i8} {
      %X_1 = riscv.mv %X : (!riscv.reg<a0>) -> !riscv.reg
      %Y_1 = riscv.mv %Y : (!riscv.reg<a1>) -> !riscv.reg
      %zero = riscv.get_register : !riscv.reg<zero>
      %zero_float = riscv.fcvt.d.w %zero : (!riscv.reg<zero>) -> !riscv.freg
      %zero_vector = riscv_snitch.vfcpka.s.s %zero_float, %zero_float : (!riscv.freg, !riscv.freg) -> !riscv.freg
      snitch_stream.streaming_region {
        patterns = [
          #snitch_stream.stride_pattern<ub = [128], strides = [8]>
        ]
      } ins(%X_1 : !riscv.reg) outs(%Y_1 : !riscv.reg) {
      ^bb0(%x : !snitch.readable<!riscv.freg<ft0>>, %0 : !snitch.writable<!riscv.freg<ft1>>):
        %c128 = riscv.li 128 : !riscv.reg
        %c0 = riscv.li 0 : !riscv.reg
        %c1 = riscv.li 1 : !riscv.reg
        riscv_scf.for %i : !riscv.reg = %c0 to %c128 step %c1 {
          %x_1 = riscv_snitch.read from %x : !riscv.freg<ft0>
          %y = riscv_snitch.vfmax.s %x_1, %zero_vector : (!riscv.freg<ft0>, !riscv.freg) -> !riscv.freg<ft1>
          riscv_snitch.write %y to %0 : !riscv.freg<ft1>
        }
      }
      riscv_func.return
    }
  }

// CHECK-NEXT:  # Regalloc stats: {"preallocated_float": ["ft0", "ft1", "ft2"], "preallocated_int": ["a0", "a1", "zero"], "allocated_float": ["ft0", "ft1", "ft3"], "allocated_int": ["a0", "a1", "t0", "t1", "t2", "zero"]}
// CHECK-NEXT:  .globl reluf32
// CHECK-NEXT:  .p2align 2
// CHECK-NEXT:  reluf32:
// CHECK-NEXT:      mv t1, a0
// CHECK-NEXT:      mv t0, a1
// CHECK-NEXT:      fcvt.d.w ft3, zero
// CHECK-NEXT:      vfcpka.s.s ft3, ft3, ft3
// CHECK-NEXT:      li t2, 127
// CHECK-NEXT:      scfgwi t2, 95                                # dm 31 dim 0 bound
// CHECK-NEXT:      li t2, 8
// CHECK-NEXT:      scfgwi t2, 223                               # dm 31 dim 0 stride
// CHECK-NEXT:      scfgwi zero, 63                              # dm 31 repeat
// CHECK-NEXT:      scfgwi t1, 768                               # dm 0 dim 0 source
// CHECK-NEXT:      scfgwi t0, 897                               # dm 1 dim 0 destination
// CHECK-NEXT:      csrrsi zero, 1984, 1                         # SSR enable
// CHECK-NEXT:      li t0, 127
// CHECK-NEXT:      frep.o t0, 1, 0, 0
// CHECK-NEXT:      vfmax.s ft1, ft0, ft3
// CHECK-NEXT:      csrrci zero, 1984, 1                         # SSR disable
// CHECK-NEXT:      ret
