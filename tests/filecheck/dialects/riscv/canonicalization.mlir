// RUN: xdsl-opt -p canonicalize %s | filecheck %s

// CHECK: %any, %a0 = "test.op"() : () -> (!riscv.reg<>, !riscv.reg<a0>)
%any, %a0 = "test.op"() : () -> (!riscv.reg<>, !riscv.reg<a0>)

// CHECK-NOT: %anymv = riscv.mv %any : (!riscv.reg<>) -> !riscv.reg<>
%anymv = riscv.mv %any : (!riscv.reg<>) -> !riscv.reg<>

// CHECK-NOT: %a0mv = riscv.mv %a0 : (!riscv.reg<a0>) -> !riscv.reg<a0>
%a0mv = riscv.mv %a0 : (!riscv.reg<a0>) -> !riscv.reg<a0>

// CHECK-NEXT: "test.op"(%any, %a0) : (!riscv.reg<>, !riscv.reg<a0>) -> ()
"test.op"(%anymv, %a0mv) : (!riscv.reg<>, !riscv.reg<a0>) -> ()
