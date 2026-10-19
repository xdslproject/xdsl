// RUN: xdsl-opt -p 'x86-set-arch{arch=avx2}' %s | filecheck %s
// RUN: xdsl-opt -p 'x86-set-arch{arch=avx512}' %s | filecheck %s --check-prefix=AVX512
// RUN: xdsl-opt -p 'x86-set-arch{arch=unknown}' %s | filecheck %s --check-prefix=UNKNOWN
// RUN: xdsl-opt -p 'x86-set-arch{arch=avx2},x86-set-arch{arch=avx512}' %s | filecheck %s --check-prefix=AVX512

// The target is recorded on the module, so passes downstream can read it
// instead of each taking their own `arch` option. Setting it twice keeps the
// last value, which is what lets a pipeline override an earlier default.

builtin.module {
  "test.op"() : () -> ()
}

// CHECK:        builtin.module attributes {x86.arch = "avx2"} {
// CHECK-NEXT:     "test.op"() : () -> ()
// CHECK-NEXT:   }

// AVX512:       builtin.module attributes {x86.arch = "avx512"} {
// UNKNOWN:      builtin.module attributes {x86.arch = "unknown"} {
