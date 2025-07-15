// RUN: xdsl-opt -p canonicalize %s | filecheck %s

// CHECK:       builtin.module {

// CHECK-NEXT:    %i0, %i1, %i2 = "test.op"() : () -> (!x86.reg<rdi>, !x86.reg<rsi>, !x86.reg)
%i0, %i1, %i2 = "test.op"() : () -> (!x86.reg<rdi>, !x86.reg<rsi>, !x86.reg)

// CHECK-NEXT:    %o1 = x86.ds.mov %i1 : (!x86.reg<rsi>) -> !x86.reg<rdx>
// CHECK-NEXT:    %o2 = x86.ds.mov %i2 : (!x86.reg) -> !x86.reg
// CHECK-NEXT:    "test.op"(%i0, %o1, %o2) : (!x86.reg<rdi>, !x86.reg<rdx>, !x86.reg) -> ()
%o0 = x86.ds.mov %i0 : (!x86.reg<rdi>) -> !x86.reg<rdi>
%o1 = x86.ds.mov %i1 : (!x86.reg<rsi>) -> !x86.reg<rdx>
%o2 = x86.ds.mov %i2 : (!x86.reg) -> !x86.reg
"test.op"(%o0, %o1, %o2) : (!x86.reg<rdi>, !x86.reg<rdx>, !x86.reg) -> ()

// CHECK-NEXT:    %zero = x86.di.mov 0 : () -> !x86.reg
// CHECK-NEXT:    "test.op"(%zero) : (!x86.reg) -> ()
%zero = x86.di.mov 0 : () -> !x86.reg
"test.op"(%zero) : (!x86.reg) -> ()

// CHECK-NEXT:    "test.op"(%i0) : (!x86.reg<rdi>) -> ()
%add_immediate_zero_reg = x86.rs.add %i0, %zero : (!x86.reg<rdi>, !x86.reg) -> !x86.reg<rdi>
"test.op"(%add_immediate_zero_reg) : (!x86.reg<rdi>) -> ()

// CHECK-NEXT:  }
