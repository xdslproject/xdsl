// RUN: xdsl-opt -p lower-func-to-riscv-func --split-input-file  %s | filecheck %s

builtin.module {
    func.func @main() {
        "test.op"() : () -> ()
        func.return
    }
}

// CHECK:       builtin.module {
// CHECK-NEXT:      riscv_func.func @main() {
// CHECK-NEXT:          "test.op"() : () -> ()
// CHECK-NEXT:          riscv_func.return
// CHECK-NEXT:      }
// CHECK-NEXT:  }
