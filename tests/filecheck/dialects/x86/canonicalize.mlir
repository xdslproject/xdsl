// RUN: xdsl-opt -p canonicalize %s | filecheck %s

// CHECK:       builtin.module {

// CHECK-NEXT:    %i0, %i1, %i2 = "test.op"() : () -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64)
%i0, %i1, %i2 = "test.op"() : () -> (!x86.reg64<rdi>, !x86.reg64<rsi>, !x86.reg64)

// CHECK-NEXT:    %o1 = x86.ds.mov %i1 : (!x86.reg64<rsi>) -> !x86.reg64<rdx>
// CHECK-NEXT:    %o2 = x86.ds.mov %i2 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:    "test.op"(%i0, %o1, %o2) : (!x86.reg64<rdi>, !x86.reg64<rdx>, !x86.reg64) -> ()
%o0 = x86.ds.mov %i0 : (!x86.reg64<rdi>) -> !x86.reg64<rdi>
%o1 = x86.ds.mov %i1 : (!x86.reg64<rsi>) -> !x86.reg64<rdx>
%o2 = x86.ds.mov %i2 : (!x86.reg64) -> !x86.reg64
"test.op"(%o0, %o1, %o2) : (!x86.reg64<rdi>, !x86.reg64<rdx>, !x86.reg64) -> ()

// Unused constants get optimized out
%c0 = x86.di.mov 0 : () -> !x86.reg64
%c0_0 = x86.ds.mov %c0 : (!x86.reg64) -> !x86.reg64
%c32 = x86.di.mov 32 : () -> !x86.reg64

// CHECK-NEXT:    %moved_i0 = x86.ds.mov %i0 : (!x86.reg64<rdi>) -> !x86.reg64<rbx>
// CHECK-NEXT:    "test.op"(%moved_i0) : (!x86.reg64<rbx>) -> ()
%moved_i0 = x86.ds.mov %i0 : (!x86.reg64<rdi>) -> !x86.reg64<rbx>
%add_immediate_zero_reg = x86.rs.add %moved_i0, %c0 : (!x86.reg64<rbx>, !x86.reg64) -> !x86.reg64<rbx>
"test.op"(%add_immediate_zero_reg) : (!x86.reg64<rbx>) -> ()

// Constant memory offsets get optimized out
%moved_i1 = x86.ds.mov %i0 : (!x86.reg64<rdi>) -> !x86.reg64
%offset_ptr = x86.rs.add %moved_i1, %c32 : (!x86.reg64, !x86.reg64) -> !x86.reg64

// CHECK-NEXT:     %rm_mov = x86.dm.mov %i0, 40 : (!x86.reg64<rdi>) -> !x86.reg64<rax>
// CHECK-NEXT:     x86.ms.mov %i0, %rm_mov, 40 : (!x86.reg64<rdi>, !x86.reg64<rax>) -> ()
%rm_mov = x86.dm.mov %offset_ptr, 8 : (!x86.reg64) -> !x86.reg64<rax>
x86.ms.mov %offset_ptr, %rm_mov, 8 : (!x86.reg64, !x86.reg64<rax>) -> ()

// CHECK-NEXT:     %avx = x86.dm.vmovupd %i0, 64 : (!x86.reg64<rdi>) -> !x86.avx2reg<ymm1>
// CHECK-NEXT:     x86.ms.vmovapd %i0, %avx, 64 : (!x86.reg64<rdi>, !x86.avx2reg<ymm1>) -> ()
%avx = x86.dm.vmovupd %offset_ptr, 32 : (!x86.reg64) -> !x86.avx2reg<ymm1>
x86.ms.vmovapd %offset_ptr, %avx, 32 : (!x86.reg64, !x86.avx2reg<ymm1>) -> ()

// Unused memory reads get eliminated
%unused_read = x86.dm.mov %i0, 8 : (!x86.reg64<rdi>) -> !x86.reg64

// CHECK-NEXT:  }
