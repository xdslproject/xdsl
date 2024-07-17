// RUN: xdsl-opt %s -p convert-riscv-to-llvm | filecheck %s

%0 = riscv.get_register : !riscv.reg<zero>
// CHECK: builtin.module {
// CHECK-NEXT: %0 = riscv.get_register : !riscv.reg<zero>


// standard risc-v instructions

%1 = riscv.li 0 : !riscv.reg
// CHECK-NEXT: %1 = "llvm.inline_asm"() <{"asm_string" = "li $0, 0", "constraints" = "=r", "asm_dialect" = 0 : i64}> : () -> i32
// CHECK-NEXT: %2 = builtin.unrealized_conversion_cast %1 : i32 to !riscv.reg

%2 = riscv.sub %1, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT: %3 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT: %4 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT: %5 = "llvm.inline_asm"(%3, %4) <{"asm_string" = "sub $0, $1, $2", "constraints" = "=r,r,r", "asm_dialect" = 0 : i64}> : (i32, i32) -> i32
// CHECK-NEXT: %6 = builtin.unrealized_conversion_cast %5 : i32 to !riscv.reg

%3 = riscv.div %1, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT: %7 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT: %8 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT: %9 = "llvm.inline_asm"(%7, %8) <{"asm_string" = "div $0, $1, $2", "constraints" = "=r,r,r", "asm_dialect" = 0 : i64}> : (i32, i32) -> i32
// CHECK-NEXT: %10 = builtin.unrealized_conversion_cast %9 : i32 to !riscv.reg

// named riscv registers:

%4 = riscv.li 0 : !riscv.reg<a0>
// CHECK-NEXT: %1 = "llvm.inline_asm"() <{"asm_string" = "li a0, 0", "constraints" = "=r", "asm_dialect" = 0 : i64}> : () -> i32
// CHECK-NEXT: %2 = builtin.unrealized_conversion_cast %1 : i32 to !riscv.reg

// standard risc-v instructions

%5 = riscv.li 0 : !riscv.reg
// CHECK-NEXT: %1 = "llvm.inline_asm"() <{"asm_string" = "li $0, 0", "constraints" = "=r", "asm_dialect" = 0 : i64}> : () -> i32
// CHECK-NEXT: %2 = builtin.unrealized_conversion_cast %1 : i32 to !riscv.reg

%6 = riscv.sub %1, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT: %3 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT: %4 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT: %5 = "llvm.inline_asm"(%3, %4) <{"asm_string" = "sub $0, $1, $2", "constraints" = "=r,r,r", "asm_dialect" = 0 : i64}> : (i32, i32) -> i32
// CHECK-NEXT: %6 = builtin.unrealized_conversion_cast %5 : i32 to !riscv.reg

%7 = riscv.div %1, %1 : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT: %7 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT: %8 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT: %9 = "llvm.inline_asm"(%7, %8) <{"asm_string" = "div $0, $1, $2", "constraints" = "=r,r,r", "asm_dialect" = 0 : i64}> : (i32, i32) -> i32
// CHECK-NEXT: %10 = builtin.unrealized_conversion_cast %9 : i32 to !riscv.reg


// csr instructions

%8 = riscv.csrrs %0, 3860, "r" : (!riscv.reg<zero>) -> !riscv.reg
// CHECK-NEXT: %11 = "llvm.inline_asm"() <{"asm_string" = "csrrs $0, 3860, x0", "constraints" = "=r", "asm_dialect" = 0 : i64}> : () -> i32
// CHECK-NEXT: %12 = builtin.unrealized_conversion_cast %11 : i32 to !riscv.reg

%9 = riscv.csrrs %0, 1986 : (!riscv.reg<zero>) -> !riscv.reg<zero>
// CHECK-NEXT: %13 = "llvm.inline_asm"() <{"asm_string" = "csrrs $0, 1986, x0", "constraints" = "=r", "asm_dialect" = 0 : i64}> : () -> i32
// CHECK-NEXT: %14 = builtin.unrealized_conversion_cast %13 : i32 to !riscv.reg<zero>

%10 = riscv.csrrci 1984, 1 : () -> !riscv.reg
// CHECK-NEXT: %15 = "llvm.inline_asm"() <{"asm_string" = "csrrci $0, 1984, 1", "constraints" = "=r", "asm_dialect" = 0 : i64}> : () -> i32
// CHECK-NEXT: %16 = builtin.unrealized_conversion_cast %15 : i32 to !riscv.reg

// custom snitch instructions

riscv_snitch.dmsrc %1, %1 : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT: %17 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT: %18 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT: "llvm.inline_asm"(%17, %18) <{"asm_string" = ".insn r 0x2b, 0, 0, x0, $0, $1", "constraints" = "r,r", "asm_dialect" = 0 : i64}> : (i32, i32) -> ()

riscv_snitch.dmdst %1, %1 : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT: %19 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT: %20 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT: "llvm.inline_asm"(%19, %20) <{"asm_string" = ".insn r 0x2b, 0, 1, x0, $0, $1", "constraints" = "r,r", "asm_dialect" = 0 : i64}> : (i32, i32) -> ()

riscv_snitch.dmstr %1, %1 : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT: %21 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT: %22 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT: "llvm.inline_asm"(%21, %22) <{"asm_string" = ".insn r 0x2b, 0, 6, x0, $0, $1", "constraints" = "r,r", "asm_dialect" = 0 : i64}> : (i32, i32) -> ()

riscv_snitch.dmrep %1 : (!riscv.reg) -> ()
// CHECK-NEXT: %23 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT: "llvm.inline_asm"(%23) <{"asm_string" = ".insn r 0x2b, 0, 7, x0, $0, x0", "constraints" = "r", "asm_dialect" = 0 : i64}> : (i32) -> ()

%11 = riscv_snitch.dmcpyi %1, 2 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT: %24 = builtin.unrealized_conversion_cast %2 : !riscv.reg to i32
// CHECK-NEXT: %25 = "llvm.inline_asm"(%24) <{"asm_string" = ".insn r 0x2b, 0, 2, $0, $1, 2", "constraints" = "=r,r", "asm_dialect" = 0 : i64}> : (i32) -> i32
// CHECK-NEXT: %26 = builtin.unrealized_conversion_cast %25 : i32 to !riscv.reg
