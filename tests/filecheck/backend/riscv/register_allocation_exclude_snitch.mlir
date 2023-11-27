// RUN: xdsl-opt --split-input-file -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive}" %s | filecheck %s
// RUN: xdsl-opt --split-input-file -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive exclude_snitch_reserved=false}" %s | filecheck %s --check-prefix=CHECK-SNITCH-UNRESERVED

riscv_func.func @main() {
  %stride_pattern, %ptr0 = "test.op"() : () -> (!snitch_stream.stride_pattern_type, !riscv.reg<>)
  %s0 = "snitch_stream.strided_read"(%ptr0, %stride_pattern) {"dm" = #builtin.int<0>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type) -> !stream.readable<!riscv.freg<>>
  %v0, %v1, %v2 = "test.op"() : () -> (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>)
  %sum1 = riscv.fadd.s %v0, %v1 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %sum2 = riscv.fadd.s %sum1, %v2 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  riscv_func.return
}

// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @main() {
// CHECK-NEXT:      %stride_pattern, %ptr0 = "test.op"() : () -> (!snitch_stream.stride_pattern_type, !riscv.reg<>)
// CHECK-NEXT:      %s0 = "snitch_stream.strided_read"(%ptr0, %stride_pattern) {"dm" = #builtin.int<0>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type) -> !stream.readable<!riscv.freg<>>
// CHECK-NEXT:      %v0, %v1, %v2 = "test.op"() : () -> (!riscv.freg<ft3>, !riscv.freg<ft5>, !riscv.freg<ft4>)
// CHECK-NEXT:      %sum1 = riscv.fadd.s %v0, %v1 : (!riscv.freg<ft3>, !riscv.freg<ft5>) -> !riscv.freg<ft3>
// CHECK-NEXT:      %sum2 = riscv.fadd.s %sum1, %v2 : (!riscv.freg<ft3>, !riscv.freg<ft4>) -> !riscv.freg<ft3>
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK-SNITCH-UNRESERVED:       builtin.module {
// CHECK-SNITCH-UNRESERVED-NEXT:    riscv_func.func @main() {
// CHECK-SNITCH-UNRESERVED-NEXT:      %stride_pattern, %ptr0 = "test.op"() : () -> (!snitch_stream.stride_pattern_type, !riscv.reg<>)
// CHECK-SNITCH-UNRESERVED-NEXT:      %s0 = "snitch_stream.strided_read"(%ptr0, %stride_pattern) {"dm" = #builtin.int<0>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type) -> !stream.readable<!riscv.freg<>>
// CHECK-SNITCH-UNRESERVED-NEXT:      %v0, %v1, %v2 = "test.op"() : () -> (!riscv.freg<ft0>, !riscv.freg<ft2>, !riscv.freg<ft1>)
// CHECK-SNITCH-UNRESERVED-NEXT:      %sum1 = riscv.fadd.s %v0, %v1 : (!riscv.freg<ft0>, !riscv.freg<ft2>) -> !riscv.freg<ft0>
// CHECK-SNITCH-UNRESERVED-NEXT:      %sum2 = riscv.fadd.s %sum1, %v2 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft0>
// CHECK-SNITCH-UNRESERVED-NEXT:      riscv_func.return
// CHECK-SNITCH-UNRESERVED-NEXT:    }
// CHECK-SNITCH-UNRESERVED-NEXT:  }

// -----

riscv_func.func @main() {
  %stride_pattern, %ptr0 = "test.op"() : () -> (!snitch_stream.stride_pattern_type, !riscv.reg<>)
  %s0 = "snitch_stream.strided_write"(%ptr0, %stride_pattern) {"dm" = #builtin.int<0>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type) -> !stream.writable<!riscv.freg<>>
  %v0, %v1, %v2 = "test.op"() : () -> (!riscv.freg<>, !riscv.freg<>, !riscv.freg<>)
  %sum1 = riscv.fadd.s %v0, %v1 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  %sum2 = riscv.fadd.s %sum1, %v2 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
  riscv_func.return
}

// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @main() {
// CHECK-NEXT:      %stride_pattern, %ptr0 = "test.op"() : () -> (!snitch_stream.stride_pattern_type, !riscv.reg<>)
// CHECK-NEXT:      %s0 = "snitch_stream.strided_write"(%ptr0, %stride_pattern) {"dm" = #builtin.int<0>, "rank" = #builtin.int<2>} : (!riscv.reg<>,  !snitch_stream.stride_pattern_type) -> !stream.writable<!riscv.freg<>>
// CHECK-NEXT:      %v0, %v1, %v2 = "test.op"() : () -> (!riscv.freg<ft3>, !riscv.freg<ft5>, !riscv.freg<ft4>)
// CHECK-NEXT:      %sum1 = riscv.fadd.s %v0, %v1 : (!riscv.freg<ft3>, !riscv.freg<ft5>) -> !riscv.freg<ft3>
// CHECK-NEXT:      %sum2 = riscv.fadd.s %sum1, %v2 : (!riscv.freg<ft3>, !riscv.freg<ft4>) -> !riscv.freg<ft3>
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK-SNITCH-UNRESERVED:       builtin.module {
// CHECK-SNITCH-UNRESERVED-NEXT:    riscv_func.func @main() {
// CHECK-SNITCH-UNRESERVED-NEXT:      %stride_pattern, %ptr0 = "test.op"() : () -> (!snitch_stream.stride_pattern_type, !riscv.reg<>)
// CHECK-SNITCH-UNRESERVED-NEXT:      %s0 = "snitch_stream.strided_write"(%ptr0, %stride_pattern) {"dm" = #builtin.int<0>, "rank" = #builtin.int<2>} : (!riscv.reg<>, !snitch_stream.stride_pattern_type) -> !stream.writable<!riscv.freg<>>
// CHECK-SNITCH-UNRESERVED-NEXT:      %v0, %v1, %v2 = "test.op"() : () -> (!riscv.freg<ft0>, !riscv.freg<ft2>, !riscv.freg<ft1>)
// CHECK-SNITCH-UNRESERVED-NEXT:      %sum1 = riscv.fadd.s %v0, %v1 : (!riscv.freg<ft0>, !riscv.freg<ft2>) -> !riscv.freg<ft0>
// CHECK-SNITCH-UNRESERVED-NEXT:      %sum2 = riscv.fadd.s %sum1, %v2 : (!riscv.freg<ft0>, !riscv.freg<ft1>) -> !riscv.freg<ft0>
// CHECK-SNITCH-UNRESERVED-NEXT:      riscv_func.return
// CHECK-SNITCH-UNRESERVED-NEXT:    }
// CHECK-SNITCH-UNRESERVED-NEXT:  }
