// RUN: xdsl-opt --split-input-file -p "riscv-allocate-registers{allocation_strategy=LivenessBlockNaive}" %s | filecheck %s

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
