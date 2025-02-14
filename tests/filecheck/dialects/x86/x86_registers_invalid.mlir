// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

// CHECK: foo not in x86 
"test.op"() { reg = !x86.reg<foo> } : () -> ()

// -----

// CHECK: foo not in AVX2
"test.op"() { reg = !x86.avx2reg<foo> } : () -> ()

// -----

// CHECK: foo not in AVX512
"test.op"() { reg = !x86.avx512reg<foo> } : () -> ()

// -----

// CHECK: foo not in SSE
"test.op"() { reg = !x86.ssereg<foo> } : () -> ()
