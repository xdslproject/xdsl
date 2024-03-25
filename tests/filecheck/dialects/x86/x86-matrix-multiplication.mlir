// RUN: xdsl-opt -t x86-asm %s | filecheck %s
%rax = x86.get_register : () -> !x86.reg<rax>
%rsi = x86.get_register : () -> !x86.reg<rsi>
%rdi = x86.get_register : () -> !x86.reg<rdi>
%rdx = x86.get_register : () -> !x86.reg<rdx>
%rcx = x86.get_register : () -> !x86.reg<rcx>

%0 = x86.rm_mov %rax, %rsi: (!x86.reg<rax>, !x86.reg<rsi>) -> !x86.reg<rax>
%1 = x86.rm_imul %0, %rdi: (!x86.reg<rax>, !x86.reg<rdi>) -> !x86.reg<rax>
%2 = x86.rm_add %1, %rdx: (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
x86.mr_mov %rdx, %2 : (!x86.reg<rdx>, !x86.reg<rax>) -> ()
%3 = x86.rm_mov %rcx, %rsi, 8: (!x86.reg<rcx>, !x86.reg<rsi>) -> !x86.reg<rcx>
%4 = x86.rm_imul %3, %rdi, 4: (!x86.reg<rcx>, !x86.reg<rdi>) -> !x86.reg<rcx>
%5 = x86.rr_add %4, %2: (!x86.reg<rcx>, !x86.reg<rax>) -> !x86.reg<rcx>
x86.mr_mov %rdx, %5 : (!x86.reg<rdx>, !x86.reg<rcx>) -> ()
%6 = x86.rm_mov %rax, %rsi, 4: (!x86.reg<rax>, !x86.reg<rsi>) -> !x86.reg<rax>
%7 = x86.rm_imul %6, %rdi: (!x86.reg<rax>, !x86.reg<rdi>) -> !x86.reg<rax>
%8 = x86.rm_add %7, %rdx, 4: (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
x86.mr_mov %rdx, %8, 4 : (!x86.reg<rdx>, !x86.reg<rax>) -> ()
%9 = x86.rm_mov %5, %rsi, 12: (!x86.reg<rcx>, !x86.reg<rsi>) -> !x86.reg<rcx>
%10 = x86.rm_imul %9, %rdi, 4: (!x86.reg<rcx>, !x86.reg<rdi>) -> !x86.reg<rcx>
%11 = x86.rr_add %10, %8: (!x86.reg<rcx>, !x86.reg<rax>) -> !x86.reg<rcx>
x86.mr_mov %rdx, %11, 4 : (!x86.reg<rdx>, !x86.reg<rcx>) -> ()
%12 = x86.rm_mov %8, %rsi: (!x86.reg<rax>, !x86.reg<rsi>) -> !x86.reg<rax>
%13 = x86.rm_imul %12, %rdi, 8: (!x86.reg<rax>, !x86.reg<rdi>) -> !x86.reg<rax>
%14 = x86.rm_add %13, %rdx, 8: (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
x86.mr_mov %rdx, %14, 8 : (!x86.reg<rdx>, !x86.reg<rax>) -> ()
%15 = x86.rm_mov %11, %rsi, 8: (!x86.reg<rcx>, !x86.reg<rsi>) -> !x86.reg<rcx>
%16 = x86.rm_imul %15, %rdi, 12: (!x86.reg<rcx>, !x86.reg<rdi>) -> !x86.reg<rcx>
%17 = x86.rr_add %16, %14: (!x86.reg<rcx>, !x86.reg<rax>) -> !x86.reg<rcx>
x86.mr_mov %rdx, %17, 8 : (!x86.reg<rdx>, !x86.reg<rcx>) -> ()
%18 = x86.rm_mov %14, %rsi, 4: (!x86.reg<rax>, !x86.reg<rsi>) -> !x86.reg<rax>
%19 = x86.rm_imul %18, %rdi, 8: (!x86.reg<rax>, !x86.reg<rdi>) -> !x86.reg<rax>
%20 = x86.rm_add %19, %rdx, 12: (!x86.reg<rax>, !x86.reg<rdx>) -> !x86.reg<rax>
x86.mr_mov %rdx, %20, 12 : (!x86.reg<rdx>, !x86.reg<rax>) -> ()
%21 = x86.rm_mov %17, %rsi, 12: (!x86.reg<rcx>, !x86.reg<rsi>) -> !x86.reg<rcx>
%22 = x86.rm_imul %21, %rdi, 12: (!x86.reg<rcx>, !x86.reg<rdi>) -> !x86.reg<rcx>
%23 = x86.rr_add %22, %20: (!x86.reg<rcx>, !x86.reg<rax>) -> !x86.reg<rcx>
x86.mr_mov %rdx, %23, 12 : (!x86.reg<rdx>, !x86.reg<rcx>) -> ()

//CHECK:     mov rax, [rsi]
//CHECK-NEXT: imul rax, [rdi]
//CHECK-NEXT: add rax, [rdx]
//CHECK-NEXT: mov [rdx], rax
//CHECK-NEXT: mov rcx, [rsi + 8]
//CHECK-NEXT: imul rcx, [rdi + 4]
//CHECK-NEXT: add rcx, rax
//CHECK-NEXT: mov [rdx], rcx
//CHECK-NEXT: mov rax, [rsi + 4]
//CHECK-NEXT: imul rax, [rdi]
//CHECK-NEXT: add rax, [rdx + 4]
//CHECK-NEXT: mov [rdx + 4], rax
//CHECK-NEXT: mov rcx, [rsi + 12]
//CHECK-NEXT: imul rcx, [rdi + 4]
//CHECK-NEXT: add rcx, rax
//CHECK-NEXT: mov [rdx + 4], rcx
//CHECK-NEXT: mov rax, [rsi]
//CHECK-NEXT: imul rax, [rdi + 8]
//CHECK-NEXT: add rax, [rdx + 8]
//CHECK-NEXT: mov [rdx + 8], rax
//CHECK-NEXT: mov rcx, [rsi + 8]
//CHECK-NEXT: imul rcx, [rdi + 12]
//CHECK-NEXT: add rcx, rax
//CHECK-NEXT: mov [rdx + 8], rcx
//CHECK-NEXT: mov rax, [rsi + 4]
//CHECK-NEXT: imul rax, [rdi + 8]
//CHECK-NEXT: add rax, [rdx + 12]
//CHECK-NEXT: mov [rdx + 12], rax
//CHECK-NEXT: mov rcx, [rsi + 12]
//CHECK-NEXT: imul rcx, [rdi + 12]
//CHECK-NEXT: add rcx, rax
//CHECK-NEXT: mov [rdx + 12], rcx