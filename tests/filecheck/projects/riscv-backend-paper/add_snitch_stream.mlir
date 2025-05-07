// RUN: xdsl-run %s | filecheck %s

builtin.module {
  riscv.assembly_section ".data" {
    riscv.label "a"
    riscv.directive ".word" "0x0,0x3ff00000,0x0,0x40000000,0x0,0x40080000,0x0,0x40100000,0x0,0x40140000,0x0,0x40180000"
    riscv.label "b"
    riscv.directive ".word" "0x0,0x0,0x0,0x3fd00000,0x0,0x3fe00000,0x0,0x3fe80000,0x0,0x3ff00000,0x0,0x3ff40000"
    riscv.label "c"
    riscv.directive ".word" "0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0"
  }
  riscv_func.func public @main() {
    %A = riscv.li "a" : !riscv.reg
    %B = riscv.li "b" : !riscv.reg
    %C = riscv.li "c" : !riscv.reg
    "snitch_stream.streaming_region"(%A, %B, %C) <{
      "stride_patterns" = [#snitch_stream.stride_pattern<ub = [2, 3], strides = [24, 8]>],
      operandSegmentSizes = array<i32: 2, 1>
    }> ({
    ^0(%a_stream : !snitch.readable<!riscv.freg<ft0>>, %b_stream : !snitch.readable<!riscv.freg<ft1>>, %c_stream : !snitch.writable<!riscv.freg<ft2>>):
        %c5 = riscv.li 5 : !riscv.reg
        riscv_snitch.frep_outer %c5 {
            %a = riscv_snitch.read from %a_stream : !riscv.freg<ft0>
            %b = riscv_snitch.read from %b_stream : !riscv.freg<ft1>
            %c = riscv.fadd.d %a, %b : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft2>
            riscv_snitch.write %c to %c_stream : !riscv.freg<ft2>
        }
    }) : (!riscv.reg, !riscv.reg, !riscv.reg) -> ()
    %5 = riscv.mv %C : (!riscv.reg) -> !riscv.reg
    %v0 = riscv.fld %5, 0 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg) -> !riscv.freg
    %v1 = riscv.fld %C, 8 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg) -> !riscv.freg
    %v2 = riscv.fld %C, 16 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg) -> !riscv.freg
    %v3 = riscv.fld %C, 24 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg) -> !riscv.freg
    %v4 = riscv.fld %C, 32 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg) -> !riscv.freg
    %v5 = riscv.fld %C, 40 {"comment" = "load double from memref of shape (2, 3)"} : (!riscv.reg) -> !riscv.freg
    riscv_debug.printf %v0, %v1, %v2, %v3, %v4, %v5 "[[{}, {}, {}], [{}, {}, {}]]" : (!riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg) -> ()
    riscv_func.return
  }
}

// CHECK{LITERAL}: [[1.0, 2.25, 3.5], [4.75, 6.0, 7.25]]
