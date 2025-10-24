// RUN: xdsl-opt %s --split-input-file --verify-diagnostics | filecheck %s

// CHECK: Invalid register name foo for register type x86.reg.
"test.op"() { reg = !x86.reg<foo> } : () -> ()

// -----

// CHECK: Invalid register name foo for register type x86.avx2reg.
"test.op"() { reg = !x86.avx2reg<foo> } : () -> ()

// -----

// CHECK: Invalid register name foo for register type x86.avx512reg.
"test.op"() { reg = !x86.avx512reg<foo> } : () -> ()

// -----

// CHECK: Invalid register name foo for register type x86.ssereg.
"test.op"() { reg = !x86.ssereg<foo> } : () -> ()

// -----

// CHECK: Invalid register name foo for register type x86.avx512maskreg.
"test.op"() { reg = !x86.avx512maskreg<foo> } : () -> ()
