// RUN: xdsl-opt -p lower-riscv-func{insert_exit_syscall=true} %s | filecheck %s

"builtin.module"() ({
// CHECK:      builtin.module {

    riscv_func.func @main() {
        riscv_func.return
    }

// CHECK-NEXT:     riscv_func.func @main() {
// CHECK-NEXT:         %{{.*}} = rv32.li 93 : !riscv.reg<a7>
// CHECK-NEXT:         riscv.ecall
// CHECK-NEXT:         riscv_func.return
// CHECK-NEXT:     }

}) : () -> ()

// CHECK-NEXT: }
