// RUN: xdsl-run %s | filecheck %s

builtin.module {
  riscv.assembly_section ".data" {
    riscv.label "a"
    riscv.directive ".word" "0x0,0x3ff00000,0x0,-0x40100000,0x0,0x0,0x0,0x40000000,0x0,-0x40000000,0x0,0x0"
  }
  riscv.assembly_section ".data" {
    riscv.label "b"
    riscv.directive ".word" "0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0"
  }
  riscv_func.func public @main() attributes {p2align = 2 : i8} {
    %A = rv32.li "a" : !riscv.reg
    %B = rv32.li "b" : !riscv.reg
    %zero = riscv.get_register : !riscv.reg<sp>
    %zero_1 = rv32.li 0 : !riscv.reg
    riscv.sw %zero, %zero_1, -4 : (!riscv.reg<sp>, !riscv.reg) -> ()
    %zero_2 = rv32.li 0 : !riscv.reg
    riscv.sw %zero, %zero_2, -8 : (!riscv.reg<sp>, !riscv.reg) -> ()
    %zero_3 = riscv.fld %zero, -8 : (!riscv.reg<sp>) -> !riscv.freg
    "snitch_stream.streaming_region"(%A, %B) <{
      "stride_patterns" = [#snitch_stream.stride_pattern<ub = [2, 3], strides = [24, 8]>],
      operandSegmentSizes = array<i32: 1, 1>
    }> ({
    ^bb0(%a_stream : !snitch.readable<!riscv.freg<ft0>>, %b_stream : !snitch.writable<!riscv.freg<ft1>>):
      %c5 = rv32.li 5 : !riscv.reg
      riscv_snitch.frep_outer %c5 {
        %a = riscv_snitch.read from %a_stream : !riscv.freg<ft0>
        %b = riscv.fmax.d %a, %zero_3 : (!riscv.freg<ft0>, !riscv.freg) -> !riscv.freg<ft1>
        riscv_snitch.write %b to %b_stream : !riscv.freg<ft1>
      }
    }) : (!riscv.reg, !riscv.reg) -> ()
    %v0 = riscv.fld %B, 0 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg) -> !riscv.freg
    %v1 = riscv.fld %B, 8 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg) -> !riscv.freg
    %v2 = riscv.fld %B, 16 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg) -> !riscv.freg
    %v3 = riscv.fld %B, 24 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg) -> !riscv.freg
    %v4 = riscv.fld %B, 32 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg) -> !riscv.freg
    %v5 = riscv.fld %B, 40 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg) -> !riscv.freg
    riscv_debug.printf %v0, %v1, %v2, %v3, %v4, %v5 "[[{}, {}, {}], [{}, {}, {}]]" : (!riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg) -> ()
    riscv_func.return
  }
}

// CHECK{LITERAL}: [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
