// RUN: xdsl-opt -p lower-riscv-func --print-op-generic %s | filecheck %s

"builtin.module"() ({
// CHECK:      "builtin.module"() ({

    %file = "riscv.li"() {"immediate" = 0 : i32} : () -> !riscv.reg<s0>
    %success = "riscv_func.syscall"(%file) {"syscall_num" = 64 : i32}: (!riscv.reg<s0>) -> !riscv.reg<s1>
// CHECK-NEXT:     %file = "riscv.li"() {"immediate" = 0 : i32} : () -> !riscv.reg<s0>
// CHECK-NEXT:     %{{.+}} = "riscv.mv"(%{{.+}}) : (!riscv.reg<s0>) -> !riscv.reg<a0>
// CHECK-NEXT:     %{{.+}} = "riscv.li"() {"immediate" = 64 : i32} : () -> !riscv.reg<a7>
// CHECK-NEXT:     "riscv.ecall"() : () -> ()
// CHECK-NEXT:     %{{.+}} = "riscv.get_register"() : () -> !riscv.reg<a0>
// CHECK-NEXT:     %{{.+}} = "riscv.mv"(%{{.+}}) : (!riscv.reg<a0>) -> !riscv.reg<s1>


    "riscv_func.syscall"() {"syscall_num" = 93 : i32} : () -> ()
// CHECK-NEXT:     %{{.+}} = "riscv.li"() {"immediate" = 93 : i32} : () -> !riscv.reg<a7>
// CHECK-NEXT:     "riscv.ecall"() : () -> ()

    "riscv_func.func"() ({
        %0 = "riscv_func.call"() {"callee" = "get_one"} : () -> !riscv.reg<x$>
        %1 = "riscv_func.call"() {"callee" = "get_one"} : () -> !riscv.reg<x$>
        %2 = "riscv_func.call"(%0, %1) {"callee" = "add"} : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
        "riscv_func.call"(%2) {"callee" = "my_print"} : (!riscv.reg<x$>) -> ()
        "riscv_func.return"() : () -> ()
    }) {"sym_name" = "main"} : () -> ()

// CHECK-NEXT:     "riscv.label"() ({
// CHECK-NEXT:         "riscv.jal"() {"immediate" = #riscv.label<"get_one">} : () -> ()
// CHECK-NEXT:         %{{.*}} = "riscv.get_register"() : () -> !riscv.reg<a0>
// CHECK-NEXT:         %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<a0>) -> !riscv.reg<x$>
// CHECK-NEXT:         "riscv.jal"() {"immediate" = #riscv.label<"get_one">} : () -> ()
// CHECK-NEXT:         %{{.*}} = "riscv.get_register"() : () -> !riscv.reg<a0>
// CHECK-NEXT:         %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<a0>) -> !riscv.reg<x$>
// CHECK-NEXT:         %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<x$>) -> !riscv.reg<a0>
// CHECK-NEXT:         %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<x$>) -> !riscv.reg<a1>
// CHECK-NEXT:         "riscv.jal"() {"immediate" = #riscv.label<"add">} : () -> ()
// CHECK-NEXT:         %{{.*}} = "riscv.get_register"() : () -> !riscv.reg<a0>
// CHECK-NEXT:         %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<a0>) -> !riscv.reg<x$>
// CHECK-NEXT:         %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<x$>) -> !riscv.reg<a0>
// CHECK-NEXT:         "riscv.jal"() {"immediate" = #riscv.label<"my_print">} : () -> ()
// CHECK-NEXT:         "riscv.ret"() : () -> ()
// CHECK-NEXT:     }) {"label" = #riscv.label<"main">} : () -> ()


    "riscv_func.func"() ({
        "riscv_func.return"() : () -> ()
    }) {"sym_name" = "my_print"} : () -> ()

// CHECK-NEXT:     "riscv.label"() ({
// CHECK-NEXT:         "riscv.ret"() : () -> ()
// CHECK-NEXT:     }) {"label" = #riscv.label<"my_print">} : () -> ()

    "riscv_func.func"() ({
        %0 = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<x$>
        "riscv_func.return"(%0) : (!riscv.reg<x$>) -> ()
    }) {"sym_name" = "get_one"} : () -> ()

// CHECK-NEXT:     "riscv.label"() ({
// CHECK-NEXT:         %{{.*}} = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<x$>
// CHECK-NEXT:         %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<x$>) -> !riscv.reg<a0>
// CHECK-NEXT:         "riscv.ret"() : () -> ()
// CHECK-NEXT:     }) {"label" = #riscv.label<"get_one">} : () -> ()

    "riscv_func.func"() ({
    ^0(%0 : !riscv.reg<x$>, %1 : !riscv.reg<x$>):
        %2 = "riscv.add"(%0, %1) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
        "riscv_func.return"(%2) : (!riscv.reg<x$>) -> ()
    }) {"sym_name" = "add"} : () -> ()

// CHECK-NEXT:     "riscv.label"() ({
// CHECK-NEXT:         %{{.*}} = "riscv.get_register"() : () -> !riscv.reg<a0>
// CHECK-NEXT:         %{{.*}} = "riscv.get_register"() : () -> !riscv.reg<a1>
// CHECK-NEXT:         %{{.*}} = "riscv.add"(%{{.*}}, %{{.*}}) : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<x$>
// CHECK-NEXT:         %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<x$>) -> !riscv.reg<a0>
// CHECK-NEXT:         "riscv.ret"() : () -> ()
// CHECK-NEXT:     }) {"label" = #riscv.label<"add">} : () -> ()

}) : () -> ()

// CHECK-NEXT: }) : () -> ()
