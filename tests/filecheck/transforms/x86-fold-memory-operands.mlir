// RUN: xdsl-opt -p 'x86-fold-memory-operands{arch=avx2}' %s | filecheck %s

// A vector load feeding an FMA folds into the memory operand. This form is
// VEX-encodable, so it applies at AVX2.
%ptr = "test.op"() : () -> !x86.reg64
%acc = "test.op"() : () -> !x86.avx2reg
%b = "test.op"() : () -> !x86.avx2reg
%load = x86.dm.vmovupd [%ptr + 8] : (!x86.reg64) -> !x86.avx2reg
%fma = x86.rss.vfmadd231pd %acc, %b, %load : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
"test.op"(%fma) : (!x86.avx2reg) -> ()
// CHECK:       %fma = x86.rsm.vfmadd231pd %acc, %b, [%ptr + 8] : (!x86.avx2reg, !x86.avx2reg, !x86.reg64) -> !x86.avx2reg
// CHECK-NOT:   x86.dm.vmovupd [%ptr + 8]

// Aligned loads fold too; the memory-operand form has no alignment requirement.
%ptr_a = "test.op"() : () -> !x86.reg64
%acc_a = "test.op"() : () -> !x86.avx2reg
%b_a = "test.op"() : () -> !x86.avx2reg
%load_a = x86.dm.vmovaps [%ptr_a] : (!x86.reg64) -> !x86.avx2reg
%fma_a = x86.rss.vfmadd231ps %acc_a, %b_a, %load_a : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
"test.op"(%fma_a) : (!x86.avx2reg) -> ()
// CHECK:       %fma_a = x86.rsm.vfmadd231ps %acc_a, %b_a, [%ptr_a] : (!x86.avx2reg, !x86.avx2reg, !x86.reg64) -> !x86.avx2reg

// The multiply operands commute, so a load in source1 folds as well, leaving
// the register operand in source1 of the fused instruction.
%ptr_c = "test.op"() : () -> !x86.reg64
%acc_c = "test.op"() : () -> !x86.avx2reg
%b_c = "test.op"() : () -> !x86.avx2reg
%load_c = x86.dm.vmovupd [%ptr_c] : (!x86.reg64) -> !x86.avx2reg
%fma_c = x86.rss.vfmadd231pd %acc_c, %load_c, %b_c : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
"test.op"(%fma_c) : (!x86.avx2reg) -> ()
// CHECK:       %fma_c = x86.rsm.vfmadd231pd %acc_c, %b_c, [%ptr_c] : (!x86.avx2reg, !x86.avx2reg, !x86.reg64) -> !x86.avx2reg

// A broadcast-load does NOT fold at AVX2: the embedded broadcast modifier
// requires EVEX encoding, which AVX2 does not have.
%ptr_y = "test.op"() : () -> !x86.reg64
%acc_y = "test.op"() : () -> !x86.avx2reg
%b_y = "test.op"() : () -> !x86.avx2reg
%bcast_y = x86.dm.vbroadcastsd [%ptr_y] : (!x86.reg64) -> !x86.avx2reg
%fma_y = x86.rss.vfmadd231pd %acc_y, %b_y, %bcast_y : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
"test.op"(%fma_y) : (!x86.avx2reg) -> ()
// CHECK:       %bcast_y = x86.dm.vbroadcastsd [%ptr_y] : (!x86.reg64) -> !x86.avx2reg
// CHECK-NEXT:  %fma_y = x86.rss.vfmadd231pd %acc_y, %b_y, %bcast_y

// A load with more than one use does not fold: doing so would duplicate the
// memory access rather than remove it.
%ptr_m = "test.op"() : () -> !x86.reg64
%acc_m = "test.op"() : () -> !x86.avx2reg
%b_m = "test.op"() : () -> !x86.avx2reg
%load_m = x86.dm.vmovupd [%ptr_m] : (!x86.reg64) -> !x86.avx2reg
%fma_m = x86.rss.vfmadd231pd %acc_m, %b_m, %load_m : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
"test.op"(%load_m) : (!x86.avx2reg) -> ()
"test.op"(%fma_m) : (!x86.avx2reg) -> ()
// CHECK:       %load_m = x86.dm.vmovupd [%ptr_m] : (!x86.reg64) -> !x86.avx2reg
// CHECK-NEXT:  %fma_m = x86.rss.vfmadd231pd %acc_m, %b_m, %load_m

// A store between the load and the FMA blocks the fold: folding would sink the
// read past a write that may alias it.
%ptr_s = "test.op"() : () -> !x86.reg64
%acc_s = "test.op"() : () -> !x86.avx2reg
%b_s = "test.op"() : () -> !x86.avx2reg
%load_s = x86.dm.vmovupd [%ptr_s] : (!x86.reg64) -> !x86.avx2reg
x86.ms.vmovupd %ptr_s, %b_s : (!x86.reg64, !x86.avx2reg) -> ()
%fma_s = x86.rss.vfmadd231pd %acc_s, %b_s, %load_s : (!x86.avx2reg, !x86.avx2reg, !x86.avx2reg) -> !x86.avx2reg
"test.op"(%fma_s) : (!x86.avx2reg) -> ()
// CHECK:       %load_s = x86.dm.vmovupd [%ptr_s] : (!x86.reg64) -> !x86.avx2reg
// CHECK:       %fma_s = x86.rss.vfmadd231pd %acc_s, %b_s, %load_s
