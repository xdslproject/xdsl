// RUN: xdsl-opt --split-input-file -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive}" %s | filecheck %s
// RUN: xdsl-opt --split-input-file -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive exclude_snitch_reserved=false}" %s | filecheck %s --check-prefix=CHECK-SNITCH-UNRESERVED

riscv_func.func @main() {
  %stream = "test.op"() : () -> (!snitch.readable<!riscv.freg<ft0>>)
  %v0, %v1, %v2 = "test.op"() : () -> (!riscv.freg, !riscv.freg, !riscv.freg)
  %read = riscv_snitch.read from %stream : !riscv.freg<ft0>
  %sum1 = riscv.fadd.s %v0, %v1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
  riscv_func.return
}

// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @main() {
// CHECK-NEXT:      %stream = "test.op"() : () -> !snitch.readable<!riscv.freg<ft0>>
// CHECK-NEXT:      %v0, %v1, %v2 = "test.op"() : () -> (!riscv.freg<ft3>, !riscv.freg<ft4>, !riscv.freg)
// CHECK-NEXT:      %read = riscv_snitch.read from %stream : !riscv.freg<ft0>
// CHECK-NEXT:      %sum1 = riscv.fadd.s %v0, %v1 : (!riscv.freg<ft3>, !riscv.freg<ft4>) -> !riscv.freg<ft3>
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK-SNITCH-UNRESERVED:       builtin.module {
// CHECK-SNITCH-UNRESERVED-NEXT:    riscv_func.func @main() {
// CHECK-SNITCH-UNRESERVED-NEXT:      %stream = "test.op"() : () -> !snitch.readable<!riscv.freg<ft0>>
// CHECK-SNITCH-UNRESERVED-NEXT:      %v0, %v1, %v2 = "test.op"() : () -> (!riscv.freg<ft1>, !riscv.freg<ft2>, !riscv.freg)
// CHECK-SNITCH-UNRESERVED-NEXT:      %read = riscv_snitch.read from %stream : !riscv.freg<ft0>
// CHECK-SNITCH-UNRESERVED-NEXT:      %sum1 = riscv.fadd.s %v0, %v1 : (!riscv.freg<ft1>, !riscv.freg<ft2>) -> !riscv.freg<ft1>
// CHECK-SNITCH-UNRESERVED-NEXT:      riscv_func.return
// CHECK-SNITCH-UNRESERVED-NEXT:    }
// CHECK-SNITCH-UNRESERVED-NEXT:  }

// -----

riscv_func.func @main() {
  %stream, %val = "test.op"() : () -> (!snitch.writable<!riscv.freg<ft0>>, !riscv.freg<ft0>)
  %v0, %v1, %v2 = "test.op"() : () -> (!riscv.freg, !riscv.freg, !riscv.freg)
  riscv_snitch.write %val to %stream : !riscv.freg<ft0>
  %sum1 = riscv.fadd.s %v0, %v1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
  riscv_func.return
}

// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func @main() {
// CHECK-NEXT:      %stream, %val = "test.op"() : () -> (!snitch.writable<!riscv.freg<ft0>>, !riscv.freg<ft0>)
// CHECK-NEXT:      %v0, %v1, %v2 = "test.op"() : () -> (!riscv.freg<ft3>, !riscv.freg<ft4>, !riscv.freg)
// CHECK-NEXT:      riscv_snitch.write %val to %stream : !riscv.freg<ft0>
// CHECK-NEXT:      %sum1 = riscv.fadd.s %v0, %v1 : (!riscv.freg<ft3>, !riscv.freg<ft4>) -> !riscv.freg<ft3>
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK-SNITCH-UNRESERVED:       builtin.module {
// CHECK-SNITCH-UNRESERVED-NEXT:    riscv_func.func @main() {
// CHECK-SNITCH-UNRESERVED-NEXT:      %stream, %val = "test.op"() : () -> (!snitch.writable<!riscv.freg<ft0>>, !riscv.freg<ft0>)
// CHECK-SNITCH-UNRESERVED-NEXT:      %v0, %v1, %v2 = "test.op"() : () -> (!riscv.freg<ft1>, !riscv.freg<ft2>, !riscv.freg)
// CHECK-SNITCH-UNRESERVED-NEXT:      riscv_snitch.write %val to %stream : !riscv.freg<ft0>
// CHECK-SNITCH-UNRESERVED-NEXT:      %sum1 = riscv.fadd.s %v0, %v1 : (!riscv.freg<ft1>, !riscv.freg<ft2>) -> !riscv.freg<ft1>
// CHECK-SNITCH-UNRESERVED-NEXT:      riscv_func.return
// CHECK-SNITCH-UNRESERVED-NEXT:    }
// CHECK-SNITCH-UNRESERVED-NEXT:  }
