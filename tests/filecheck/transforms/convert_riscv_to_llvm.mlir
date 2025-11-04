// RUN: xdsl-opt %s -p convert-riscv-to-llvm | filecheck %s
// RUN: xdsl-opt %s -p convert-riscv-to-llvm,reconcile-unrealized-casts,dce | filecheck %s --check-prefix COMPACT


%reg = riscv.li 0 : !riscv.reg
%a0 = riscv.li 0 : !riscv.reg<a0>
%x0 = riscv.get_register : !riscv.reg<zero>

// CHECK: builtin.module {
// CHECK-NEXT:  %reg = "llvm.inline_asm"() <{asm_string = "li $0, 0", constraints = "=r", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : () -> i32
// CHECK-NEXT:  %reg_1 = builtin.unrealized_conversion_cast %reg : i32 to !riscv.reg
// CHECK-NEXT:  %a0 = "llvm.inline_asm"() <{asm_string = "li $0, 0", constraints = "=r", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : () -> i32
// CHECK-NEXT:  %a0_1 = builtin.unrealized_conversion_cast %a0 : i32 to !riscv.reg<a0>
// CHECK-NEXT:  %x0 = riscv.get_register : !riscv.reg<zero>

// standard risc-v instructions

%li = riscv.li 0 : !riscv.reg
// CHECK-NEXT:  %li = "llvm.inline_asm"() <{asm_string = "li $0, 0", constraints = "=r", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : () -> i32
// CHECK-NEXT:  %li_1 = builtin.unrealized_conversion_cast %li : i32 to !riscv.reg

%sub = riscv.sub %reg, %reg : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:  %sub = builtin.unrealized_conversion_cast %reg_1 : !riscv.reg to i32
// CHECK-NEXT:  %sub_1 = builtin.unrealized_conversion_cast %reg_1 : !riscv.reg to i32
// CHECK-NEXT:  %sub_2 = "llvm.inline_asm"(%sub, %sub_1) <{asm_string = "sub $0, $1, $2", constraints = "=r,rI,rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32, i32) -> i32
// CHECK-NEXT:  %sub_3 = builtin.unrealized_conversion_cast %sub_2 : i32 to !riscv.reg

%div = riscv.div %reg, %reg : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:  %div = builtin.unrealized_conversion_cast %reg_1 : !riscv.reg to i32
// CHECK-NEXT:  %div_1 = builtin.unrealized_conversion_cast %reg_1 : !riscv.reg to i32
// CHECK-NEXT:  %div_2 = "llvm.inline_asm"(%div, %div_1) <{asm_string = "div $0, $1, $2", constraints = "=r,rI,rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32, i32) -> i32
// CHECK-NEXT:  %div_3 = builtin.unrealized_conversion_cast %div_2 : i32 to !riscv.reg

// named riscv registers:

%li_named = riscv.li 0 : !riscv.reg<a0>
// CHECK-NEXT:  %li_named = "llvm.inline_asm"() <{asm_string = "li $0, 0", constraints = "=r", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : () -> i32
// CHECK-NEXT:  %li_named_1 = builtin.unrealized_conversion_cast %li_named : i32 to !riscv.reg<a0>

%sub_named = riscv.sub %a0, %a0 : (!riscv.reg<a0>, !riscv.reg<a0>) -> !riscv.reg
// CHECK-NEXT:  %sub_named = builtin.unrealized_conversion_cast %a0_1 : !riscv.reg<a0> to i32
// CHECK-NEXT:  %sub_named_1 = builtin.unrealized_conversion_cast %a0_1 : !riscv.reg<a0> to i32
// CHECK-NEXT:  %sub_named_2 = "llvm.inline_asm"(%sub_named, %sub_named_1) <{asm_string = "sub $0, $1, $2", constraints = "=r,rI,rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32, i32) -> i32
// CHECK-NEXT:  %sub_named_3 = builtin.unrealized_conversion_cast %sub_named_2 : i32 to !riscv.reg

%div_named = riscv.div %a0, %a0 : (!riscv.reg<a0>, !riscv.reg<a0>) -> !riscv.reg
// CHECK-NEXT:  %div_named = builtin.unrealized_conversion_cast %a0_1 : !riscv.reg<a0> to i32
// CHECK-NEXT:  %div_named_1 = builtin.unrealized_conversion_cast %a0_1 : !riscv.reg<a0> to i32
// CHECK-NEXT:  %div_named_2 = "llvm.inline_asm"(%div_named, %div_named_1) <{asm_string = "div $0, $1, $2", constraints = "=r,rI,rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32, i32) -> i32
// CHECK-NEXT:  %div_named_3 = builtin.unrealized_conversion_cast %div_named_2 : i32 to !riscv.reg


// csr instructions

%csrss = riscv.csrrs %x0, 3860, "r" : (!riscv.reg<zero>) -> !riscv.reg
// CHECK-NEXT:  %csrss = "llvm.inline_asm"() <{asm_string = "csrrs $0, 3860, x0", constraints = "=r", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : () -> i32
// CHECK-NEXT:  %csrss_1 = builtin.unrealized_conversion_cast %csrss : i32 to !riscv.reg

%csrrs = riscv.csrrs %x0, 1986 : (!riscv.reg<zero>) -> !riscv.reg<zero>
// CHECK-NEXT:  %csrrs = riscv.get_register : !riscv.reg<zero>
// CHECK-NEXT:  "llvm.inline_asm"() <{asm_string = "csrrs x0, 1986, x0", constraints = "", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()

%csrrci = riscv.csrrci 1984, 1 : () -> !riscv.reg
// CHECK-NEXT:  %csrrci = "llvm.inline_asm"() <{asm_string = "csrrci $0, 1984, 1", constraints = "=r", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : () -> i32
// CHECK-NEXT:  %csrrci_1 = builtin.unrealized_conversion_cast %csrrci : i32 to !riscv.reg


// custom snitch instructions

riscv_snitch.dmsrc %reg, %reg : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %reg_1 : !riscv.reg to i32
// CHECK-NEXT:  %1 = builtin.unrealized_conversion_cast %reg_1 : !riscv.reg to i32
// CHECK-NEXT:  "llvm.inline_asm"(%0, %1) <{asm_string = ".insn r 0x2b, 0, 0, x0, $0, $1", constraints = "rI,rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32, i32) -> ()

riscv_snitch.dmdst %reg, %reg : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:  %2 = builtin.unrealized_conversion_cast %reg_1 : !riscv.reg to i32
// CHECK-NEXT:  %3 = builtin.unrealized_conversion_cast %reg_1 : !riscv.reg to i32
// CHECK-NEXT:  "llvm.inline_asm"(%2, %3) <{asm_string = ".insn r 0x2b, 0, 1, x0, $0, $1", constraints = "rI,rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32, i32) -> ()

riscv_snitch.dmstr %reg, %reg : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:  %4 = builtin.unrealized_conversion_cast %reg_1 : !riscv.reg to i32
// CHECK-NEXT:  %5 = builtin.unrealized_conversion_cast %reg_1 : !riscv.reg to i32
// CHECK-NEXT:  "llvm.inline_asm"(%4, %5) <{asm_string = ".insn r 0x2b, 0, 6, x0, $0, $1", constraints = "rI,rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32, i32) -> ()

riscv_snitch.dmrep %reg : (!riscv.reg) -> ()
// CHECK-NEXT:  %6 = builtin.unrealized_conversion_cast %reg_1 : !riscv.reg to i32
// CHECK-NEXT:  "llvm.inline_asm"(%6) <{asm_string = ".insn r 0x2b, 0, 7, x0, $0, x0", constraints = "rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32) -> ()

%dmcpyi = riscv_snitch.dmcpyi %reg, 2 : (!riscv.reg) -> !riscv.reg
// CHECK-NEXT:  %dmcpyi = builtin.unrealized_conversion_cast %reg_1 : !riscv.reg to i32
// CHECK-NEXT:  %dmcpyi_1 = "llvm.inline_asm"(%dmcpyi) <{asm_string = ".insn r 0x2b, 0, 2, $0, $1, 2", constraints = "=r,rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32) -> i32
// CHECK-NEXT:  %dmcpyi_2 = builtin.unrealized_conversion_cast %dmcpyi_1 : i32 to !riscv.reg


// ------------------------------------------------------- //
// compact representation after reconciling casts and DCE: //
// ------------------------------------------------------- //

// COMPACT:      builtin.module {
// COMPACT-NEXT:   %reg = "llvm.inline_asm"() <{asm_string = "li $0, 0", constraints = "=r", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : () -> i32
// COMPACT-NEXT:   %a0 = "llvm.inline_asm"() <{asm_string = "li $0, 0", constraints = "=r", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : () -> i32
// COMPACT-NEXT:   %li = "llvm.inline_asm"() <{asm_string = "li $0, 0", constraints = "=r", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : () -> i32
// COMPACT-NEXT:   %sub = "llvm.inline_asm"(%reg, %reg) <{asm_string = "sub $0, $1, $2", constraints = "=r,rI,rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32, i32) -> i32
// COMPACT-NEXT:   %div = "llvm.inline_asm"(%reg, %reg) <{asm_string = "div $0, $1, $2", constraints = "=r,rI,rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32, i32) -> i32
// COMPACT-NEXT:   %li_named = "llvm.inline_asm"() <{asm_string = "li $0, 0", constraints = "=r", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : () -> i32
// COMPACT-NEXT:   %sub_named = "llvm.inline_asm"(%a0, %a0) <{asm_string = "sub $0, $1, $2", constraints = "=r,rI,rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32, i32) -> i32
// COMPACT-NEXT:   %div_named = "llvm.inline_asm"(%a0, %a0) <{asm_string = "div $0, $1, $2", constraints = "=r,rI,rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32, i32) -> i32
// COMPACT-NEXT:   %csrss = "llvm.inline_asm"() <{asm_string = "csrrs $0, 3860, x0", constraints = "=r", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : () -> i32
// COMPACT-NEXT:   "llvm.inline_asm"() <{asm_string = "csrrs x0, 1986, x0", constraints = "", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
// COMPACT-NEXT:   %csrrci = "llvm.inline_asm"() <{asm_string = "csrrci $0, 1984, 1", constraints = "=r", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : () -> i32
// COMPACT-NEXT:   "llvm.inline_asm"(%reg, %reg) <{asm_string = ".insn r 0x2b, 0, 0, x0, $0, $1", constraints = "rI,rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32, i32) -> ()
// COMPACT-NEXT:   "llvm.inline_asm"(%reg, %reg) <{asm_string = ".insn r 0x2b, 0, 1, x0, $0, $1", constraints = "rI,rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32, i32) -> ()
// COMPACT-NEXT:   "llvm.inline_asm"(%reg, %reg) <{asm_string = ".insn r 0x2b, 0, 6, x0, $0, $1", constraints = "rI,rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32, i32) -> ()
// COMPACT-NEXT:   "llvm.inline_asm"(%reg) <{asm_string = ".insn r 0x2b, 0, 7, x0, $0, x0", constraints = "rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32) -> ()
// COMPACT-NEXT:   %dmcpyi = "llvm.inline_asm"(%reg) <{asm_string = ".insn r 0x2b, 0, 2, $0, $1, 2", constraints = "=r,rI", asm_dialect = 0 : i64, tail_call_kind = #llvm.tailcallkind<none>}> : (i32) -> i32
// COMPACT-NEXT: }
