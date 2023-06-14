// RUN: xdsl-opt -p cf-to-riscv %s | filecheck %s

"builtin.module"() ({
  // CHECK:      builtin.module {
  func.func @cond_br1() {
    %cond = "arith.constant"() {"value" = true} : () -> i1
    "cf.cond_br"(%cond) [^bb1, ^bb2] {"operand_segment_sizes" = array<i32: 1, 0, 0>} : (i1) -> ()
  ^bb1:
    func.return
  ^bb2:
    func.return
  }
}) : () -> ()

// CHECK-NEXT: }
