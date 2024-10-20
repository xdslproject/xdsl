// RUN: xdsl-opt %s -p convert-riscv-to-llvm | filecheck %s
// RUN: xdsl-opt %s -p convert-riscv-to-llvm,reconcile-unrealized-casts,dce | filecheck %s --check-prefix COMPACT


%0 = riscv.get_register : !riscv.reg<zero>
// CHECK: builtin.module {
// CHECK-NEXT:  %0 = riscv.get_register : !riscv.reg<zero>


// standard risc-v instructions

%1 = riscv.li 0 : !riscv.reg
// CHECK-NEXT:  %1 = "llvm.inline_asm"() <{"asm_string" = "li $0, 0", "constraints" = "=r", "asm_dialect" = 0 : i64}> : () -> i32
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %1 : i32 to !riscv.reg

%2 = riscv.sub %1, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT:  %5 = "llvm.inline_asm"(%3, %4) <{"asm_string" = "sub $0, $1, $2", "constraints" = "=r,rI,rI", "asm_dialect" = 0 : i64}> : (i32, i32) -> i32
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %5 : i32 to !riscv.reg

%3 = riscv.div %1, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:  %7 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT:  %8 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT:  %9 = "llvm.inline_asm"(%7, %8) <{"asm_string" = "div $0, $1, $2", "constraints" = "=r,rI,rI", "asm_dialect" = 0 : i64}> : (i32, i32) -> i32
// CHECK-NEXT:  %10 = builtin.unrealized_conversion_cast %9 : i32 to !riscv.reg


// named riscv registers:

%4 = riscv.li 0 : !riscv.reg<a0>
// CHECK-NEXT:  %11 = "llvm.inline_asm"() <{"asm_string" = "li $0, 0", "constraints" = "=r", "asm_dialect" = 0 : i64}> : () -> i32
// CHECK-NEXT:  %12 = builtin.unrealized_conversion_cast %11 : i32 to !riscv.reg<a0>

%5 = riscv.sub %4, %4 : (!riscv.reg<a0>, !riscv.reg<a0>) -> !riscv.reg
// CHECK-NEXT:  %13 = builtin.unrealized_conversion_cast %12 : !riscv.reg<a0> to i32
// CHECK-NEXT:  %14 = builtin.unrealized_conversion_cast %12 : !riscv.reg<a0> to i32
// CHECK-NEXT:  %15 = "llvm.inline_asm"(%13, %14) <{"asm_string" = "sub $0, $1, $2", "constraints" = "=r,rI,rI", "asm_dialect" = 0 : i64}> : (i32, i32) -> i32
// CHECK-NEXT:  %16 = builtin.unrealized_conversion_cast %15 : i32 to !riscv.reg

%6 = riscv.div %4, %4 : (!riscv.reg<a0>, !riscv.reg<a0>) -> !riscv.reg
// CHECK-NEXT:  %17 = builtin.unrealized_conversion_cast %12 : !riscv.reg<a0> to i32
// CHECK-NEXT:  %18 = builtin.unrealized_conversion_cast %12 : !riscv.reg<a0> to i32
// CHECK-NEXT:  %19 = "llvm.inline_asm"(%17, %18) <{"asm_string" = "div $0, $1, $2", "constraints" = "=r,rI,rI", "asm_dialect" = 0 : i64}> : (i32, i32) -> i32
// CHECK-NEXT:  %20 = builtin.unrealized_conversion_cast %19 : i32 to !riscv.reg


// csr instructions

%7 = riscv.csrrs %0, 3860, "r" : (!riscv.reg<zero>) -> !riscv.reg
// CHECK-NEXT:  %21 = "llvm.inline_asm"() <{"asm_string" = "csrrs $0, 3860, x0", "constraints" = "=r", "asm_dialect" = 0 : i64}> : () -> i32
// CHECK-NEXT:  %22 = builtin.unrealized_conversion_cast %21 : i32 to !riscv.reg

%8 = riscv.csrrs %0, 1986 : (!riscv.reg<zero>) -> !riscv.reg<zero>
// CHECK-NEXT:  %23 = riscv.get_register : !riscv.reg<zero>
// CHECK-NEXT:  "llvm.inline_asm"() <{"asm_string" = "csrrs x0, 1986, x0", "constraints" = "", "asm_dialect" = 0 : i64}> : () -> ()

%9 = riscv.csrrci 1984, 1 : () -> !riscv.reg
// CHECK-NEXT:  %24 = "llvm.inline_asm"() <{"asm_string" = "csrrci $0, 1984, 1", "constraints" = "=r", "asm_dialect" = 0 : i64}> : () -> i32
// CHECK-NEXT:  %25 = builtin.unrealized_conversion_cast %24 : i32 to !riscv.reg


// custom snitch instructions

riscv_snitch.dmsrc %1, %1 : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:  %26 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT:  %27 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT:  "llvm.inline_asm"(%26, %27) <{"asm_string" = ".insn r 0x2b, 0, 0, x0, $0, $1", "constraints" = "rI,rI", "asm_dialect" = 0 : i64}> : (i32, i32) -> ()

riscv_snitch.dmdst %1, %1 : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:  %28 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT:  %29 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT:  "llvm.inline_asm"(%28, %29) <{"asm_string" = ".insn r 0x2b, 0, 1, x0, $0, $1", "constraints" = "rI,rI", "asm_dialect" = 0 : i64}> : (i32, i32) -> ()

riscv_snitch.dmstr %1, %1 : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:  %30 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT:  %31 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT:  "llvm.inline_asm"(%30, %31) <{"asm_string" = ".insn r 0x2b, 0, 6, x0, $0, $1", "constraints" = "rI,rI", "asm_dialect" = 0 : i64}> : (i32, i32) -> ()

riscv_snitch.dmrep %1 : (!riscv.reg) -> ()
// CHECK-NEXT:  %32 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT:  "llvm.inline_asm"(%32) <{"asm_string" = ".insn r 0x2b, 0, 7, x0, $0, x0", "constraints" = "rI", "asm_dialect" = 0 : i64}> : (i32) -> ()

%10 = riscv_snitch.dmcpyi %1, 2 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:  %33 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT:  %34 = "llvm.inline_asm"(%33) <{"asm_string" = ".insn r 0x2b, 0, 2, $0, $1, 2", "constraints" = "=r,rI", "asm_dialect" = 0 : i64}> : (i32) -> i32
// CHECK-NEXT:  %35 = builtin.unrealized_conversion_cast %34 : i32 to !riscv.reg


// ------------------------------------------------------- //
// compact representation after reconciling casts and DCE: //
// ------------------------------------------------------- //

// COMPACT:      builtin.module {
// COMPACT-NEXT:   %0 = "llvm.inline_asm"() <{"asm_string" = "li $0, 0", "constraints" = "=r", "asm_dialect" = 0 : i64}> : () -> i32
// COMPACT-NEXT:   %1 = "llvm.inline_asm"(%0, %0) <{"asm_string" = "sub $0, $1, $2", "constraints" = "=r,rI,rI", "asm_dialect" = 0 : i64}> : (i32, i32) -> i32
// COMPACT-NEXT:   %2 = "llvm.inline_asm"(%0, %0) <{"asm_string" = "div $0, $1, $2", "constraints" = "=r,rI,rI", "asm_dialect" = 0 : i64}> : (i32, i32) -> i32
// COMPACT-NEXT:   %3 = "llvm.inline_asm"() <{"asm_string" = "li $0, 0", "constraints" = "=r", "asm_dialect" = 0 : i64}> : () -> i32
// COMPACT-NEXT:   %4 = "llvm.inline_asm"(%3, %3) <{"asm_string" = "sub $0, $1, $2", "constraints" = "=r,rI,rI", "asm_dialect" = 0 : i64}> : (i32, i32) -> i32
// COMPACT-NEXT:   %5 = "llvm.inline_asm"(%3, %3) <{"asm_string" = "div $0, $1, $2", "constraints" = "=r,rI,rI", "asm_dialect" = 0 : i64}> : (i32, i32) -> i32
// COMPACT-NEXT:   %6 = "llvm.inline_asm"() <{"asm_string" = "csrrs $0, 3860, x0", "constraints" = "=r", "asm_dialect" = 0 : i64}> : () -> i32
// COMPACT-NEXT:   "llvm.inline_asm"() <{"asm_string" = "csrrs x0, 1986, x0", "constraints" = "", "asm_dialect" = 0 : i64}> : () -> ()
// COMPACT-NEXT:   %7 = "llvm.inline_asm"() <{"asm_string" = "csrrci $0, 1984, 1", "constraints" = "=r", "asm_dialect" = 0 : i64}> : () -> i32
// COMPACT-NEXT:   "llvm.inline_asm"(%0, %0) <{"asm_string" = ".insn r 0x2b, 0, 0, x0, $0, $1", "constraints" = "rI,rI", "asm_dialect" = 0 : i64}> : (i32, i32) -> ()
// COMPACT-NEXT:   "llvm.inline_asm"(%0, %0) <{"asm_string" = ".insn r 0x2b, 0, 1, x0, $0, $1", "constraints" = "rI,rI", "asm_dialect" = 0 : i64}> : (i32, i32) -> ()
// COMPACT-NEXT:   "llvm.inline_asm"(%0, %0) <{"asm_string" = ".insn r 0x2b, 0, 6, x0, $0, $1", "constraints" = "rI,rI", "asm_dialect" = 0 : i64}> : (i32, i32) -> ()
// COMPACT-NEXT:   "llvm.inline_asm"(%0) <{"asm_string" = ".insn r 0x2b, 0, 7, x0, $0, x0", "constraints" = "rI", "asm_dialect" = 0 : i64}> : (i32) -> ()
// COMPACT-NEXT:   %8 = "llvm.inline_asm"(%0) <{"asm_string" = ".insn r 0x2b, 0, 2, $0, $1, 2", "constraints" = "=r,rI", "asm_dialect" = 0 : i64}> : (i32) -> i32
// COMPACT-NEXT: }
