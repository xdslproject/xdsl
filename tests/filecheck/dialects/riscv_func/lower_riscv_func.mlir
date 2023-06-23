// RUN: xdsl-opt -p lower-riscv-func %s | filecheck %s

"builtin.module"() ({
// CHECK:      builtin.module {

    %file = "riscv.li"() {"immediate" = 0 : i32} : () -> !riscv.ireg<s0>
    %success = "riscv_func.syscall"(%file) {"syscall_num" = 64 : i32}: (!riscv.ireg<s0>) -> !riscv.ireg<s1>
// CHECK-NEXT:     %file = "riscv.li"() {"immediate" = 0 : i32} : () -> !riscv.ireg<s0>
// CHECK-NEXT:     %{{.+}} = "riscv.mv"(%{{.+}}) : (!riscv.ireg<s0>) -> !riscv.ireg<a0>
// CHECK-NEXT:     %{{.+}} = "riscv.li"() {"immediate" = 64 : i32} : () -> !riscv.ireg<a7>
// CHECK-NEXT:     "riscv.ecall"() : () -> ()
// CHECK-NEXT:     %{{.+}} = "riscv.get_integer_register"() : () -> !riscv.ireg<a0>
// CHECK-NEXT:     %{{.+}} = "riscv.mv"(%{{.+}}) : (!riscv.ireg<a0>) -> !riscv.ireg<s1>


    "riscv_func.syscall"() {"syscall_num" = 93 : i32} : () -> ()
// CHECK-NEXT:     %{{.+}} = "riscv.li"() {"immediate" = 93 : i32} : () -> !riscv.ireg<a7>
// CHECK-NEXT:     "riscv.ecall"() : () -> ()

    "riscv_func.func"() ({
        %0 = "riscv_func.call"() {"func_name" = "get_one"} : () -> !riscv.ireg<>
        %1 = "riscv_func.call"() {"func_name" = "get_one"} : () -> !riscv.ireg<>
        %2 = "riscv_func.call"(%0, %1) {"func_name" = "add"} : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
        "riscv_func.call"(%2) {"func_name" = "my_print"} : (!riscv.ireg<>) -> ()
        "riscv_func.return"() : () -> ()
    }) {"func_name" = "main"} : () -> ()

// CHECK-NEXT:     "riscv.label"() ({
// CHECK-NEXT:         "riscv.jal"() {"immediate" = #riscv.label<"get_one">} : () -> ()
// CHECK-NEXT:         %{{.*}} = "riscv.get_integer_register"() : () -> !riscv.ireg<a0>
// CHECK-NEXT:         %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.ireg<a0>) -> !riscv.ireg<>
// CHECK-NEXT:         "riscv.jal"() {"immediate" = #riscv.label<"get_one">} : () -> ()
// CHECK-NEXT:         %{{.*}} = "riscv.get_integer_register"() : () -> !riscv.ireg<a0>
// CHECK-NEXT:         %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.ireg<a0>) -> !riscv.ireg<>
// CHECK-NEXT:         %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.ireg<>) -> !riscv.ireg<a0>
// CHECK-NEXT:         %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.ireg<>) -> !riscv.ireg<a1>
// CHECK-NEXT:         "riscv.jal"() {"immediate" = #riscv.label<"add">} : () -> ()
// CHECK-NEXT:         %{{.*}} = "riscv.get_integer_register"() : () -> !riscv.ireg<a0>
// CHECK-NEXT:         %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.ireg<a0>) -> !riscv.ireg<>
// CHECK-NEXT:         %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.ireg<>) -> !riscv.ireg<a0>
// CHECK-NEXT:         "riscv.jal"() {"immediate" = #riscv.label<"my_print">} : () -> ()
// CHECK-NEXT:         "riscv.ret"() : () -> ()
// CHECK-NEXT:     }) {"label" = #riscv.label<"main">} : () -> ()


    "riscv_func.func"() ({
        "riscv_func.return"() : () -> ()
    }) {"func_name" = "my_print"} : () -> ()

// CHECK-NEXT:     "riscv.label"() ({
// CHECK-NEXT:         "riscv.ret"() : () -> ()
// CHECK-NEXT:     }) {"label" = #riscv.label<"my_print">} : () -> ()

    "riscv_func.func"() ({
        %0 = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.ireg<>
        "riscv_func.return"(%0) : (!riscv.ireg<>) -> ()
    }) {"func_name" = "get_one"} : () -> ()

// CHECK-NEXT:     "riscv.label"() ({
// CHECK-NEXT:         %{{.*}} = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.ireg<>
// CHECK-NEXT:         %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.ireg<>) -> !riscv.ireg<a0>
// CHECK-NEXT:         "riscv.ret"() : () -> ()
// CHECK-NEXT:     }) {"label" = #riscv.label<"get_one">} : () -> ()

    "riscv_func.func"() ({
    ^0(%0 : !riscv.ireg<>, %1 : !riscv.ireg<>):
        %2 = "riscv.add"(%0, %1) : (!riscv.ireg<>, !riscv.ireg<>) -> !riscv.ireg<>
        "riscv_func.return"(%2) : (!riscv.ireg<>) -> ()
    }) {"func_name" = "add"} : () -> ()

// CHECK-NEXT:     "riscv.label"() ({
// CHECK-NEXT:         %{{.*}} = "riscv.get_integer_register"() : () -> !riscv.ireg<a0>
// CHECK-NEXT:         %{{.*}} = "riscv.get_integer_register"() : () -> !riscv.ireg<a1>
// CHECK-NEXT:         %{{.*}} = "riscv.add"(%{{.*}}, %{{.*}}) : (!riscv.ireg<a0>, !riscv.ireg<a1>) -> !riscv.ireg<>
// CHECK-NEXT:         %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.ireg<>) -> !riscv.ireg<a0>
// CHECK-NEXT:         "riscv.ret"() : () -> ()
// CHECK-NEXT:     }) {"label" = #riscv.label<"add">} : () -> ()

}) : () -> ()

// CHECK-NEXT: }
