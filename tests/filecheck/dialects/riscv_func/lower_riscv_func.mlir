// RUN: xdsl-opt -p lower-riscv-func %s | filecheck %s

"builtin.module"() ({
// CHECK:      builtin.module {

    %file = rv32.li 0 : !riscv.reg<s0>
    %success = "riscv_func.syscall"(%file) {"syscall_num" = 64 : i32}: (!riscv.reg<s0>) -> !riscv.reg<s1>
// CHECK-NEXT:     %file = rv32.li 0 : !riscv.reg<s0>
// CHECK-NEXT:     %{{.+}} = riscv.mv %{{.+}} : (!riscv.reg<s0>) -> !riscv.reg<a0>
// CHECK-NEXT:     %{{.+}} = rv32.li 64 : !riscv.reg<a7>
// CHECK-NEXT:     riscv.ecall
// CHECK-NEXT:     %{{.+}} = riscv.get_register : !riscv.reg<a0>
// CHECK-NEXT:     %{{.+}} = riscv.mv %{{.+}} : (!riscv.reg<a0>) -> !riscv.reg<s1>


    "riscv_func.syscall"() {"syscall_num" = 93 : i32} : () -> ()
// CHECK-NEXT:     %{{.+}} = rv32.li 93 : !riscv.reg<a7>
// CHECK-NEXT:     riscv.ecall

    riscv_func.func @main() {
        %0 = riscv_func.call @get_one() : () -> !riscv.reg
        %1 = riscv_func.call @get_one() : () -> !riscv.reg
        %2 = riscv_func.call @add(%0, %1) : (!riscv.reg, !riscv.reg) -> !riscv.reg
        riscv_func.call @my_print(%2) : (!riscv.reg) -> ()
        riscv_func.return
    }

// CHECK-NEXT:    riscv_func.func @main() {
// CHECK-NEXT:        %{{.*}} = riscv_func.call @get_one() : () -> !riscv.reg
// CHECK-NEXT:        %{{.*}} = riscv_func.call @get_one() : () -> !riscv.reg
// CHECK-NEXT:        %{{.*}} = riscv_func.call @add(%{{.*}}, %{{.*}}) : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:        riscv_func.call @my_print(%{{.*}}) : (!riscv.reg) -> ()
// CHECK-NEXT:        riscv_func.return
// CHECK-NEXT:    }


    riscv_func.func @my_print() {
        riscv_func.return
    }

// CHECK-NEXT:   riscv_func.func @my_print() {
// CHECK-NEXT:       riscv_func.return
// CHECK-NEXT:   }

    riscv_func.func @get_one() {
        %0 = rv32.li 1 : !riscv.reg
        riscv_func.return %0 : !riscv.reg
    }

// CHECK-NEXT:   riscv_func.func @get_one() {
// CHECK-NEXT:       %{{\d+}} = rv32.li 1 : !riscv.reg
// CHECK-NEXT:       riscv_func.return %{{\d+}} : !riscv.reg
// CHECK-NEXT:   }

    riscv_func.func @add(%arg0 : !riscv.reg, %arg1 : !riscv.reg) {
        %res = riscv.add %arg0, %arg1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
        riscv_func.return %res : !riscv.reg
    }

// CHECK-NEXT:   riscv_func.func @add(%arg0 : !riscv.reg, %arg1 : !riscv.reg) {
// CHECK-NEXT:       %res = riscv.add %arg0, %arg1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:       riscv_func.return %res : !riscv.reg
// CHECK-NEXT:   }

    riscv_func.func private @visibility_private() {
        riscv_func.return
    }

// CHECK-NEXT:   riscv_func.func private @visibility_private() {
// CHECK-NEXT:       riscv_func.return
// CHECK-NEXT:   }

    riscv_func.func public @visibility_public() {
        riscv_func.return
    }

// CHECK-NEXT:   riscv_func.func public @visibility_public() {
// CHECK-NEXT:       riscv_func.return
// CHECK-NEXT:   }

}) : () -> ()

// CHECK-NEXT: }
