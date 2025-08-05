// RUN: xdsl-opt -p x86-infer-broadcast %s | filecheck %s

%ptr = "test.op"() : () -> !x86.reg
%s = x86.dm.mov %ptr, 0 : (!x86.reg) -> !x86.reg
// CHECK:   %r = x86.dm.vbroadcastsd %ptr, 0 : (!x86.reg) -> !x86.avx2reg
%r = x86.ds.vpbroadcastq %s : (!x86.reg) -> !x86.avx2reg
"test.op"(%r) : (!x86.avx2reg) -> ()
