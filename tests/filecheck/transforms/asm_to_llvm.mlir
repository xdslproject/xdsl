// RUN: xdsl-opt -p asm-to-llvm --split-input-file --verify-diagnostics %s | filecheck %s

// Success: empty asm.region converts to empty inline asm.
asm.region {
  asm.yield
} : () -> ()

// CHECK: llvm.inline_asm has_side_effects asm_dialect = intel "", "" : () -> ()

// -----

// Success: single-block x86-only region with allocated IO registers converts.
%v = "test.op"() : () -> i64
%res = asm.region(%v) {
^bb0(%arg0: !x86.reg64<rax>):
  %sum = x86.rs.add %arg0, %arg0 : (!x86.reg64<rax>, !x86.reg64<rax>) -> !x86.reg64<rax>
  asm.yield %sum : !x86.reg64<rax>
} : (i64) -> i64

// CHECK: %res = llvm.inline_asm has_side_effects asm_dialect = intel "add rax, rax", "+{rax}" %v : (i64) -> i64

// -----

// Success: clobber constraints include used non-boundary registers.
%v2 = "test.op"() : () -> i64
%res2 = asm.region(%v2) {
^bb0(%arg0: !x86.reg64<rax>):
  %tmp = x86.get_register : !x86.reg64<rbx>
  %sum = x86.rs.add %arg0, %tmp : (!x86.reg64<rax>, !x86.reg64<rbx>) -> !x86.reg64<rax>
  asm.yield %sum : !x86.reg64<rax>
} : (i64) -> i64

// CHECK: %res2 = llvm.inline_asm has_side_effects asm_dialect = intel "add rax, rbx", "+{rax},~{rbx}" %v2 : (i64) -> i64

// -----

// Success: boundary overlap without read+write remains separate input/output constraints.
%v3 = "test.op"() : () -> i64
%res3 = asm.region(%v3) {
^bb0(%arg0: !x86.reg64<rax>):
  asm.yield %arg0 : !x86.reg64<rax>
} : (i64) -> i64

// CHECK: %res3 = llvm.inline_asm has_side_effects asm_dialect = intel "", "={rax},{rax}" %v3 : (i64) -> i64

// -----

// Rejection: non-x86 op in region body.
%v1 = "test.op"() : () -> i64
%r1 = asm.region(%v1) {
^bb0(%arg0: !x86.reg64<rax>):
  %tmp = asm.from_reg %arg0 : !x86.reg64<rax> -> i64
  %tmp_r = asm.to_reg %tmp : i64 -> !x86.reg64<rax>
  asm.yield %tmp_r : !x86.reg64<rax>
} : (i64) -> i64

// CHECK: asm-to-llvm supports only x86 ops in asm.region body, got asm.from_reg

// -----

// Rejection: unallocated entry register in boundary constraints.
%v4 = "test.op"() : () -> i64
%r3 = asm.region(%v4) {
^bb0(%arg0: !x86.reg64):
  asm.yield %arg0 : !x86.reg64
} : (i64) -> i64

// CHECK: asm-to-llvm requires allocated x86 registers in asm.yield operands

// -----

// Rejection: only up to one yielded value is currently supported.
%v5, %v6 = "test.op"() : () -> (i64, i64)
%r5, %r6 = asm.region(%v5, %v6) {
^bb0(%arg0: !x86.reg64<rax>, %arg1: !x86.reg64<rbx>):
  asm.yield %arg0, %arg1 : !x86.reg64<rax>, !x86.reg64<rbx>
} : (i64, i64) -> (i64, i64)

// CHECK: asm-to-llvm currently supports only up to one yielded value

// -----

// Rejection: multi-block asm.region is unsupported.
%v7 = "test.op"() : () -> i64
%r4 = asm.region(%v7) {
^bb0(%arg0: !x86.reg64<rax>):
  x86.fallthrough ^bb1()
^bb1(%arg1: !x86.reg64<rax>):
  asm.yield %arg1 : !x86.reg64<rax>
} : (i64) -> i64

// CHECK: asm-to-llvm currently supports only single-block asm.region
