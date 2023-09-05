// RUN: xdsl-opt -p lower-riscv-func{insert_exit_syscall=true} %s | filecheck %s

"builtin.module"() ({
// CHECK:      builtin.module {

    riscv_func.func @main() {
        "riscv_func.return"() : () -> ()
    }

// CHECK-NEXT:   riscv.assembly_section ".text" {
// CHECK-NEXT:     riscv.directive ".globl" "main" : () -> ()
// CHECK-NEXT:     riscv.directive ".p2align" "2" : () -> ()
// CHECK-NEXT:     riscv.label "main" ({
// CHECK-NEXT:         %{{.*}} = riscv.li 93 : () -> !riscv.reg<a7>
// CHECK-NEXT:         riscv.ecall : () -> ()
// CHECK-NEXT:         riscv.ret : () -> ()
// CHECK-NEXT:     }) : () -> ()
// CHECK-NEXT:   }

}) : () -> ()

// CHECK-NEXT: }
