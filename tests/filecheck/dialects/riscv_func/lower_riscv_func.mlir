// RUN: xdsl-opt -p lower-riscv-func %s | filecheck %s

"builtin.module"() ({
// CHECK:      builtin.module {

    "riscv_func.func"() ({
        %0 = "riscv_func.call"() {"func_name" = "get_one"} : () -> !riscv.reg<>
        %1 = "riscv_func.call"() {"func_name" = "get_one"} : () -> !riscv.reg<>
        %2 = "riscv_func.call"(%0, %1) {"func_name" = "add"} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "riscv_func.call"(%2) {"func_name" = "my_print"} : () -> !riscv.reg<>
        "riscv_func.return"() : () -> ()
    }) {"func_name" = "main"} : () -> ()

// CHECK-NEXT:     "riscv.label"() {"label" = #riscv.label<"main">} : () -> ()
// CHECK-NEXT:     "riscv.jal"() {"immediate" = #riscv.label<"get_one">, "rd" = !riscv.reg<ra>} : () -> ()
// CHECK-NEXT:     %{{.*}} = "riscv.get_register"() : () -> !riscv.reg<a0>
// CHECK-NEXT:     %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<a0>) -> !riscv.reg<>
// CHECK-NEXT:     "riscv.jal"() {"immediate" = #riscv.label<"get_one">, "rd" = !riscv.reg<ra>} : () -> ()
// CHECK-NEXT:     %{{.*}} = "riscv.get_register"() : () -> !riscv.reg<a0>
// CHECK-NEXT:     %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<a0>) -> !riscv.reg<>
// CHECK-NEXT:     %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:     %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<>) -> !riscv.reg<a1>
// CHECK-NEXT:     "riscv.jal"() {"immediate" = #riscv.label<"add">, "rd" = !riscv.reg<ra>} : () -> ()
// CHECK-NEXT:     %{{.*}} = "riscv.get_register"() : () -> !riscv.reg<a0>
// CHECK-NEXT:     %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<a0>) -> !riscv.reg<>
// CHECK-NEXT:     %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:     "riscv.jal"() {"immediate" = #riscv.label<"my_print">, "rd" = !riscv.reg<ra>} : () -> ()
// CHECK-NEXT:     "riscv.ret"() : () -> ()


    "riscv_func.func"() ({
        "riscv_func.return"() : () -> ()
    }) {"func_name" = "my_print"} : () -> ()

// CHECK-NEXT:     "riscv.label"() {"label" = #riscv.label<"my_print">} : () -> ()
// CHECK-NEXT:     "riscv.ret"() : () -> ()

    "riscv_func.func"() ({
        %0 = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<>
        "riscv_func.return"(%0) : () -> ()
    }) {"func_name" = "get_one"} : () -> ()

// CHECK-NEXT:     "riscv.label"() {"label" = #riscv.label<"get_one">} : () -> ()
// CHECK-NEXT:     %{{.*}} = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<>
// CHECK-NEXT:     %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:     "riscv.ret"() : () -> ()

    "riscv_func.func"() ({
    ^0(%0 : !riscv.reg<>, %1 : !riscv.reg<>):
        %2 = "riscv.add"(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "riscv_func.return"(%2) : (!riscv.reg<>) -> ()
    }) {"func_name" = "add"} : () -> ()

// CHECK-NEXT:     "riscv.label"() {"label" = #riscv.label<"add">} : () -> ()
// CHECK-NEXT:     %{{.*}} = "riscv.get_register"() : () -> !riscv.reg<a0>
// CHECK-NEXT:     %{{.*}} = "riscv.get_register"() : () -> !riscv.reg<a1>
// CHECK-NEXT:     %{{.*}} = "riscv.add"(%{{.*}}, %{{.*}}) : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<>
// CHECK-NEXT:     %{{.*}} = "riscv.mv"(%{{.*}}) : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:     "riscv.ret"() : () -> ()

}) : () -> ()

// CHECK-NEXT: }
