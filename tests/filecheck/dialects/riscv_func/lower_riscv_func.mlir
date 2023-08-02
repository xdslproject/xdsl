// RUN: xdsl-opt -p lower-riscv-func %s | filecheck %s

"builtin.module"() ({
// CHECK:      builtin.module {

    %file = "riscv.li"() {"immediate" = 0 : i32} : () -> !riscv.reg<s0>
    %success = "riscv_func.syscall"(%file) {"syscall_num" = 64 : i32}: (!riscv.reg<s0>) -> !riscv.reg<s1>
// CHECK-NEXT:     %file = riscv.li {"immediate" = 0 : i32} : () -> !riscv.reg<s0>
// CHECK-NEXT:     %{{.+}} = riscv.mv %{{.+}} : (!riscv.reg<s0>) -> !riscv.reg<a0>
// CHECK-NEXT:     %{{.+}} = riscv.li {"immediate" = 64 : i32} : () -> !riscv.reg<a7>
// CHECK-NEXT:     riscv.ecall : () -> ()
// CHECK-NEXT:     %{{.+}} = riscv.get_register : () -> !riscv.reg<a0>
// CHECK-NEXT:     %{{.+}} = riscv.mv %{{.+}} : (!riscv.reg<a0>) -> !riscv.reg<s1>


    "riscv_func.syscall"() {"syscall_num" = 93 : i32} : () -> ()
// CHECK-NEXT:     %{{.+}} = riscv.li {"immediate" = 93 : i32} : () -> !riscv.reg<a7>
// CHECK-NEXT:     riscv.ecall : () -> ()

    "riscv_func.func"() ({
        %0 = "riscv_func.call"() {"callee" = "get_one"} : () -> !riscv.reg<>
        %1 = "riscv_func.call"() {"callee" = "get_one"} : () -> !riscv.reg<>
        %2 = "riscv_func.call"(%0, %1) {"callee" = "add"} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "riscv_func.call"(%2) {"callee" = "my_print"} : (!riscv.reg<>) -> ()
        "riscv_func.return"() : () -> ()
    }) {"sym_name" = "main"} : () -> ()

// CHECK-NEXT:     riscv.code_section ({
// CHECK-NEXT:         riscv.label {"label" = #riscv.label<"main">} : () -> ()
// CHECK-NEXT:         riscv.jal "get_one" : () -> ()
// CHECK-NEXT:         %{{.*}} = riscv.get_register : () -> !riscv.reg<a0>
// CHECK-NEXT:         %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<a0>) -> !riscv.reg<>
// CHECK-NEXT:         riscv.jal "get_one" : () -> ()
// CHECK-NEXT:         %{{.*}} = riscv.get_register : () -> !riscv.reg<a0>
// CHECK-NEXT:         %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<a0>) -> !riscv.reg<>
// CHECK-NEXT:         %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:         %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<>) -> !riscv.reg<a1>
// CHECK-NEXT:         riscv.jal "add" : () -> ()
// CHECK-NEXT:         %{{.*}} = riscv.get_register : () -> !riscv.reg<a0>
// CHECK-NEXT:         %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<a0>) -> !riscv.reg<>
// CHECK-NEXT:         %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:         riscv.jal "my_print" : () -> ()
// CHECK-NEXT:         riscv.ret : () -> ()
// CHECK-NEXT:     }) : () -> ()


    "riscv_func.func"() ({
        "riscv_func.return"() : () -> ()
    }) {"sym_name" = "my_print"} : () -> ()

// CHECK-NEXT:     riscv.code_section ({
// CHECK-NEXT:         riscv.label {"label" = #riscv.label<"my_print">} : () -> ()
// CHECK-NEXT:         riscv.ret : () -> ()
// CHECK-NEXT:     }) : () -> ()

    "riscv_func.func"() ({
        %0 = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<>
        "riscv_func.return"(%0) : (!riscv.reg<>) -> ()
    }) {"sym_name" = "get_one"} : () -> ()

// CHECK-NEXT:     riscv.code_section ({
// CHECK-NEXT:         riscv.label {"label" = #riscv.label<"get_one">} : () -> ()
// CHECK-NEXT:         %{{.*}} = riscv.li {"immediate" = 1 : i32} : () -> !riscv.reg<>
// CHECK-NEXT:         %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:         riscv.ret : () -> ()
// CHECK-NEXT:     }) : () -> ()

    "riscv_func.func"() ({
    ^0(%0 : !riscv.reg<>, %1 : !riscv.reg<>):
        %2 = "riscv.add"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "riscv_func.return"(%2) : (!riscv.reg<>) -> ()
    }) {"sym_name" = "add"} : () -> ()


// CHECK-NEXT:     riscv.code_section ({
// CHECK-NEXT:         riscv.label {"label" = #riscv.label<"add">} : () -> ()
// CHECK-NEXT:         %{{.*}} = riscv.get_register : () -> !riscv.reg<a0>
// CHECK-NEXT:         %{{.*}} = riscv.get_register : () -> !riscv.reg<a1>
// CHECK-NEXT:         %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<>
// CHECK-NEXT:         %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:         riscv.ret : () -> ()
// CHECK-NEXT:     }) : () -> ()

}) : () -> ()

// CHECK-NEXT: }
