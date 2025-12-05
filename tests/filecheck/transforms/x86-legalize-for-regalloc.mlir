// RUN: xdsl-opt -p x86-legalize-for-regalloc %s --split-input-file | filecheck %s

%reg0 = "test.op"() : () -> !x86.reg
%reg1 = x86.ds.mov %reg0: (!x86.reg) -> !x86.reg
%reg3 = "test.op"(%reg1) : (!x86.reg) -> !x86.reg

// CHECK:      builtin.module {
// CHECK-NEXT:  %reg0 = "test.op"() : () -> !x86.reg
// CHECK-NEXT:  %reg3 = "test.op"(%reg0) : (!x86.reg) -> !x86.reg
// CHECK-NEXT: }

// -----

%reg0 = "test.op"() : () -> !x86.reg
%reg1 = x86.ds.mov %reg0: (!x86.reg) -> !x86.reg
%reg3 = "test.op"(%reg0) : (!x86.reg) -> !x86.reg
%reg4 = "test.op"(%reg1) : (!x86.reg) -> !x86.reg

// CHECK:      builtin.module {
// CHECK-NEXT:  %reg0 = "test.op"() : () -> !x86.reg
// CHECK-NEXT:  %reg1 = x86.ds.mov %reg0 : (!x86.reg) -> !x86.reg
// CHECK-NEXT:  %reg3 = "test.op"(%reg0) : (!x86.reg) -> !x86.reg
// CHECK-NEXT:  %reg4 = "test.op"(%reg1) : (!x86.reg) -> !x86.reg
// CHECK-NEXT: }
