// RUN: xdsl-opt -p 'x86-fold-memory-operands{arch=avx512}' %s | filecheck %s

// On AVX-512 a broadcast-load folds into the FMA as an EVEX embedded
// broadcast, removing the broadcast instruction and the vector register it
// occupied.
%ptr = "test.op"() : () -> !x86.reg64
%acc = "test.op"() : () -> !x86.avx512reg
%b = "test.op"() : () -> !x86.avx512reg
%bcast = x86.dm.vbroadcastsd [%ptr + 16] : (!x86.reg64) -> !x86.avx512reg
%fma = x86.rss.vfmadd231pd %acc, %b, %bcast : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
"test.op"(%fma) : (!x86.avx512reg) -> ()
// CHECK:       %fma = x86.rsm.vfmadd231pd %acc, %b, [%ptr + 16] {broadcast} : (!x86.avx512reg, !x86.avx512reg, !x86.reg64) -> !x86.avx512reg
// CHECK-NOT:   x86.dm.vbroadcastsd [%ptr + 16]

// AVX512VL extends EVEX to 256-bit operands, so a ymm broadcast folds on an
// AVX-512 target as well.
%ptr_v = "test.op"() : () -> !x86.reg64
%acc_v = "test.op"() : () -> !x86.avx2reg
%b_v = "test.op"() : () -> !x86.avx2reg
%bcast_v = x86.dm.vbroadcastsd [%ptr_v] : (!x86.reg64) -> !x86.avx2reg
%fma_v = x86.rss.vfmadd231pd %acc_v, %b_v, %bcast_v : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
"test.op"(%fma_v) : (!x86.avx2reg) -> ()
// CHECK:       %fma_v = x86.rsm.vfmadd231pd %acc_v, %b_v, [%ptr_v] {broadcast} : (!x86.avx2reg, !x86.avx2reg, !x86.reg64) -> !x86.avx2reg

// Single-precision broadcasts fold to the ps form.
%ptr_s = "test.op"() : () -> !x86.reg64
%acc_s = "test.op"() : () -> !x86.avx512reg
%b_s = "test.op"() : () -> !x86.avx512reg
%bcast_s = x86.dm.vbroadcastss [%ptr_s + 4] : (!x86.reg64) -> !x86.avx512reg
%fma_s = x86.rss.vfmadd231ps %acc_s, %b_s, %bcast_s : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
"test.op"(%fma_s) : (!x86.avx512reg) -> ()
// CHECK:       %fma_s = x86.rsm.vfmadd231ps %acc_s, %b_s, [%ptr_s + 4] {broadcast} : (!x86.avx512reg, !x86.avx512reg, !x86.reg64) -> !x86.avx512reg

// Plain vector loads still fold without the broadcast modifier.
%ptr_p = "test.op"() : () -> !x86.reg64
%acc_p = "test.op"() : () -> !x86.avx512reg
%b_p = "test.op"() : () -> !x86.avx512reg
%load_p = x86.dm.vmovupd [%ptr_p] : (!x86.reg64) -> !x86.avx512reg
%fma_p = x86.rss.vfmadd231pd %acc_p, %b_p, %load_p : (!x86.avx512reg, !x86.avx512reg, !x86.avx512reg) -> !x86.avx512reg
"test.op"(%fma_p) : (!x86.avx512reg) -> ()
// CHECK:       %fma_p = x86.rsm.vfmadd231pd %acc_p, %b_p, [%ptr_p] : (!x86.avx512reg, !x86.avx512reg, !x86.reg64) -> !x86.avx512reg
