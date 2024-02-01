// RUN: xdsl-opt %s -p riscv-get-zero-register | filecheck %s

%0 = riscv.li 0 : () -> !riscv.reg<>
// CHECK:       %{{.*}} = riscv.get_register : () -> !riscv.reg<zero>
// CHECK-NEXT:  %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<zero>) -> !riscv.reg<>

%1 = riscv.li 0 : () -> !riscv.reg<zero>
// CHECK-NEXT:       %{{.*}} = riscv.get_register : () -> !riscv.reg<zero>

%2 = riscv.li 0 : () -> !riscv.reg<a0>
// CHECK:       %{{.*}} = riscv.get_register : () -> !riscv.reg<zero>
// CHECK-NEXT:  %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<zero>) -> !riscv.reg<a0>

%3 = riscv.li 1 : () -> !riscv.reg<>
// CHECK-NEXT:       %{{.*}} = riscv.li 1 : () -> !riscv.reg<>

%4 = riscv.li 1 : () -> !riscv.reg<zero>
// CHECK-NEXT:       %{{.*}} = riscv.li 1 : () -> !riscv.reg<zero>

%5 = riscv.li 1 : () -> !riscv.reg<a0>
// CHECK-NEXT:       %{{.*}} = riscv.li 1 : () -> !riscv.reg<a0>
