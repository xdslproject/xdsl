// RUN: xdsl-run %s --args "dense<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<2x3xf64>, dense<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<2x3xf64>" --verbose | filecheck %s


builtin.module {
  riscv.assembly_section ".text" {
    riscv.directive ".globl" "main"
    riscv.directive ".p2align" "2"
    riscv_func.func @main(%0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>) -> !riscv.reg<a0> {
      %2 = riscv.mv %0 : (!riscv.reg<a0>) -> !riscv.reg<>
      %3 = riscv.mv %1 : (!riscv.reg<a1>) -> !riscv.reg<>
      %4 = riscv.li 48 {"comment" = "memref alloc size"} : () -> !riscv.reg<>
      %5 = riscv.mv %4 : (!riscv.reg<>) -> !riscv.reg<a0>
      %6 = riscv_func.call @malloc(%5) : (!riscv.reg<a0>) -> !riscv.reg<a0>
      %7 = riscv.mv %6 : (!riscv.reg<a0>) -> !riscv.reg<>
      "snitch_stream.streaming_region"(%2, %3, %7) <{"stride_patterns" = [#snitch_stream.stride_pattern<ub = [6], strides = [8]>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
      ^0(%8 : !stream.readable<!riscv.freg<>>, %9 : !stream.readable<!riscv.freg<>>, %10 : !stream.writable<!riscv.freg<>>):
        %11 = riscv.li 2 : () -> !riscv.reg<>
        %12 = riscv.li 3 : () -> !riscv.reg<>
        %13 = riscv.get_register : () -> !riscv.reg<zero>
        %14 = riscv.mv %13 : (!riscv.reg<zero>) -> !riscv.reg<>
        %15 = riscv.li 1 : () -> !riscv.reg<>
        riscv_scf.for %16 : !riscv.reg<> = %14 to %11 step %15 {
          riscv_scf.for %17 : !riscv.reg<> = %14 to %12 step %15 {
            %18 = riscv_snitch.read from %8 : !riscv.freg<>
            %19 = riscv_snitch.read from %9 : !riscv.freg<>
            %20 = riscv.fsub.d %18, %19 fastmath<fast> : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
            riscv_snitch.write %20 to %10 : !riscv.freg<>
          }
        }
      }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()
      %21 = riscv.mv %7 : (!riscv.reg<>) -> !riscv.reg<a0>
      riscv_func.return %21 : !riscv.reg<a0>
    }
  }
  riscv_func.func private @malloc(!riscv.reg<a0>) -> !riscv.reg<a0>
}
