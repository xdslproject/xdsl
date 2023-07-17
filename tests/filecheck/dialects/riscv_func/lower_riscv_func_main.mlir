// RUN: xdsl-opt -p lower-riscv-func{insert_exit_syscall=true} %s | filecheck %s

"builtin.module"() ({
// CHECK:      builtin.module {

    "riscv_func.func"() ({
        "riscv_func.return"() : () -> ()
    }) {"sym_name" = "main"} : () -> ()

// CHECK-NEXT:     riscv.label "main" ({
// CHECK-NEXT:         %{{.*}} = riscv.li 93 : -> a7 |
// CHECK-NEXT:         riscv.ecall : -> |
// CHECK-NEXT:         riscv.ret : -> |
// CHECK-NEXT:     }) : -> |

}) : () -> ()

// CHECK-NEXT: }
