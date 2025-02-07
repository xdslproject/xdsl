// RUN: xdsl-opt %s --split-input-file --verify-diagnostics --parsing-diagnostics | filecheck %s

// CHECK: foo not in x86 
"test.op"() { reg = !x86.reg<foo> } : () -> ()

// -----
// CHECK: foo not in x86AVX
"test.op"() { reg = !x86.avxreg<foo> } : () -> ()
