// RUN: xdsl-opt -p x86-legalize-for-regalloc %s --split-input-file | filecheck %s

%reg0 = "test.op"() : () -> !x86.reg64
%reg1 = x86.ds.mov %reg0: (!x86.reg64) -> !x86.reg64
%reg3 = "test.op"(%reg1) : (!x86.reg64) -> !x86.reg64

// CHECK:      builtin.module {
// CHECK-NEXT:  %reg0 = "test.op"() : () -> !x86.reg64
// CHECK-NEXT:  %reg3 = "test.op"(%reg0) : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT: }

// -----

%reg0 = "test.op"() : () -> !x86.reg64
%reg1 = x86.ds.mov %reg0: (!x86.reg64) -> !x86.reg64
%reg3 = "test.op"(%reg0) : (!x86.reg64) -> !x86.reg64
%reg4 = "test.op"(%reg1) : (!x86.reg64) -> !x86.reg64

// CHECK:      builtin.module {
// CHECK-NEXT:  %reg0 = "test.op"() : () -> !x86.reg64
// CHECK-NEXT:  %reg1 = x86.ds.mov %reg0 : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:  %reg3 = "test.op"(%reg0) : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT:  %reg4 = "test.op"(%reg1) : (!x86.reg64) -> !x86.reg64
// CHECK-NEXT: }
