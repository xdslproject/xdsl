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

// Unused constants get optimized out
%c0 = x86.di.mov 0 : () -> !x86.reg
%c0_0 = x86.ds.mov %c0 : (!x86.reg) -> !x86.reg
%c32 = x86.di.mov 32 : () -> !x86.reg

// CHECK-NEXT:    %moved_i0 = x86.ds.mov %i0 : (!x86.reg<rdi>) -> !x86.reg<rbx>
// CHECK-NEXT:    "test.op"(%moved_i0) : (!x86.reg<rbx>) -> ()
%moved_i0 = x86.ds.mov %i0 : (!x86.reg<rdi>) -> !x86.reg<rbx>
%add_immediate_zero_reg = x86.rs.add %moved_i0, %c0 : (!x86.reg<rbx>, !x86.reg) -> !x86.reg<rbx>
"test.op"(%add_immediate_zero_reg) : (!x86.reg<rbx>) -> ()

// Constant memory offsets get optimized out
%moved_i1 = x86.ds.mov %i0 : (!x86.reg<rdi>) -> !x86.reg
%offset_ptr = x86.rs.add %moved_i1, %c32 : (!x86.reg, !x86.reg) -> !x86.reg

// CHECK-NEXT:     %rm_mov = x86.dm.mov %i0, 40 : (!x86.reg<rdi>) -> !x86.reg<rax>
// CHECK-NEXT:     x86.ms.mov %i0, %rm_mov, 40 : (!x86.reg<rdi>, !x86.reg<rax>) -> ()
%rm_mov = x86.dm.mov %offset_ptr, 8 : (!x86.reg) -> !x86.reg<rax>
x86.ms.mov %offset_ptr, %rm_mov, 8 : (!x86.reg, !x86.reg<rax>) -> ()

// CHECK-NEXT:     %avx = x86.dm.vmovupd %i0, 64 : (!x86.reg<rdi>) -> !x86.avx2reg<ymm1>
// CHECK-NEXT:     x86.ms.vmovapd %i0, %avx, 64 : (!x86.reg<rdi>, !x86.avx2reg<ymm1>) -> ()
%avx = x86.dm.vmovupd %offset_ptr, 32 : (!x86.reg) -> !x86.avx2reg<ymm1>
x86.ms.vmovapd %offset_ptr, %avx, 32 : (!x86.reg, !x86.avx2reg<ymm1>) -> ()

// Unused memory reads get eliminated
%unused_read = x86.dm.mov %i0, 8 : (!x86.reg<rdi>) -> !x86.reg

// CHECK-NEXT:  }
