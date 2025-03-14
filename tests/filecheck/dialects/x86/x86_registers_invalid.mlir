// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

// CHECK: Invalid register name foo for register set x86.
"test.op"() { reg = !x86.reg<foo> } : () -> ()

// -----

// CHECK: Invalid register name foo for register set AVX2.
"test.op"() { reg = !x86.avx2reg<foo> } : () -> ()

// -----

// CHECK: Invalid register name foo for register set AVX512.
"test.op"() { reg = !x86.avx512reg<foo> } : () -> ()

// -----

// CHECK: Invalid register name foo for register set SSE.
"test.op"() { reg = !x86.ssereg<foo> } : () -> ()
