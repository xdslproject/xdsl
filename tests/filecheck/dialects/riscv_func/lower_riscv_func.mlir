// RUN: xdsl-opt -p lower-riscv-func %s | filecheck %s

"builtin.module"() ({
// CHECK:      builtin.module {

    %file = riscv.li 0 : () -> !riscv.reg<s0>
    %success = "riscv_func.syscall"(%file) {"syscall_num" = 64 : i32}: (!riscv.reg<s0>) -> !riscv.reg<s1>
// CHECK-NEXT:     %file = riscv.li 0 : () -> !riscv.reg<s0>
// CHECK-NEXT:     %{{.+}} = riscv.mv %{{.+}} : (!riscv.reg<s0>) -> !riscv.reg<a0>
// CHECK-NEXT:     %{{.+}} = riscv.li 64 : () -> !riscv.reg<a7>
// CHECK-NEXT:     riscv.ecall : () -> ()
// CHECK-NEXT:     %{{.+}} = riscv.get_register : () -> !riscv.reg<a0>
// CHECK-NEXT:     %{{.+}} = riscv.mv %{{.+}} : (!riscv.reg<a0>) -> !riscv.reg<s1>


    "riscv_func.syscall"() {"syscall_num" = 93 : i32} : () -> ()
// CHECK-NEXT:     %{{.+}} = riscv.li 93 : () -> !riscv.reg<a7>
// CHECK-NEXT:     riscv.ecall : () -> ()

    riscv_func.func @main() {
        %0 = riscv_func.call @get_one() : () -> !riscv.reg<>
        %1 = riscv_func.call @get_one() : () -> !riscv.reg<>
        %2 = riscv_func.call @add(%0, %1) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        riscv_func.call @my_print(%2) : (!riscv.reg<>) -> ()
        riscv_func.return
    }

// CHECK-NEXT:     riscv.label "main" ({
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


    riscv_func.func @my_print() {
        riscv_func.return
    }

// CHECK-NEXT:     riscv.label "my_print" ({
// CHECK-NEXT:         riscv.ret : () -> ()
// CHECK-NEXT:     }) : () -> ()

    riscv_func.func @get_one() {
        %0 = riscv.li 1 : () -> !riscv.reg<>
        riscv_func.return %0 : !riscv.reg<>
    }

// CHECK-NEXT:     riscv.label "get_one" ({
// CHECK-NEXT:         %{{.*}} = riscv.li 1 : () -> !riscv.reg<>
// CHECK-NEXT:         %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:         riscv.ret : () -> ()
// CHECK-NEXT:     }) : () -> ()

    riscv_func.func @add(%0 : !riscv.reg<>, %1 : !riscv.reg<>) {
        %2 = riscv.add %0, %1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        riscv_func.return %2 : !riscv.reg<>
    }

// CHECK-NEXT:     riscv.label "add" ({
// CHECK-NEXT:         %{{.*}} = riscv.get_register : () -> !riscv.reg<a0>
// CHECK-NEXT:         %{{.*}} = riscv.get_register : () -> !riscv.reg<a1>
// CHECK-NEXT:         %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg<>
// CHECK-NEXT:         %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:         riscv.ret : () -> ()
// CHECK-NEXT:     }) : () -> ()

}) : () -> ()

// CHECK-NEXT: }
