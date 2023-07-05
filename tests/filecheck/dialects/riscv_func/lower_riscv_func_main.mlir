// RUN: xdsl-opt -p lower-riscv-func{insert_exit_syscall=true} %s | filecheck %s

"builtin.module"() ({
// CHECK:      builtin.module {

    "riscv_func.func"() ({
        "riscv_func.return"() : () -> ()
    }) {"sym_name" = "main"} : () -> ()

// CHECK-NEXT:     "riscv.label"() ({
// CHECK-NEXT:         %{{.*}} = "riscv.li"() {"immediate" = 93 : i32} : () -> !riscv.reg<a7>
// CHECK-NEXT:         "riscv.ecall"() : () -> ()
// CHECK-NEXT:         "riscv.ret"() : () -> ()
// CHECK-NEXT:     }) {"label" = #riscv.label<"main">} : () -> ()

}) : () -> ()

// CHECK-NEXT: }
