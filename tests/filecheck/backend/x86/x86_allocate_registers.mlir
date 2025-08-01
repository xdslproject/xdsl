// RUN: xdsl-opt -p x86-allocate-registers %s | filecheck %s

// CHECK:       builtin.module {

// CHECK-NEXT:    x86_func.func @external() -> ()
x86_func.func @external() -> ()


// CHECK-NEXT:    x86_func.func @main() {
// CHECK-NEXT:      %ymm0, %ymm1, %ymm2, %ymm3 = "test.op"() : () -> (!x86.avx2reg<ymm0>, !x86.avx2reg<ymm1>, !x86.avx2reg<ymm2>, !x86.avx2reg<ymm3>)
x86_func.func @main() {
  %ymm0, %ymm1, %ymm2, %ymm3 = "test.op"() : () -> (!x86.avx2reg<ymm0>, !x86.avx2reg<ymm1>, !x86.avx2reg<ymm2>, !x86.avx2reg<ymm3>)

// inout is allocated to same register
// CHECK-NEXT:      %r0 = x86.rss.vfmadd231pd %ymm0, %ymm1, %ymm2 : (!x86.avx2reg<ymm0>, !x86.avx2reg<ymm1>, !x86.avx2reg<ymm2>) -> !x86.avx2reg<ymm0>
  %r0 = x86.rss.vfmadd231pd %ymm0, %ymm1, %ymm2 : (!x86.avx2reg<ymm0>, !x86.avx2reg<ymm1>, !x86.avx2reg<ymm2>) -> !x86.avx2reg

// read-only is allocated to new register
// CHECK-NEXT:      %u0 = "test.op"() : () -> !x86.avx2reg<ymm4>
// CHECK-NEXT:      %r1 = x86.rss.vfmadd231pd %ymm3, %u0, %ymm2 : (!x86.avx2reg<ymm3>, !x86.avx2reg<ymm4>, !x86.avx2reg<ymm2>) -> !x86.avx2reg<ymm3>
  %u0 = "test.op"() : () -> !x86.avx2reg
  %r1 = x86.rss.vfmadd231pd %ymm3, %u0, %ymm2 : (!x86.avx2reg<ymm3>, !x86.avx2reg, !x86.avx2reg<ymm2>) -> !x86.avx2reg

// CHECK-NEXT:      x86_func.ret
  x86_func.ret
}

// CHECK-NEXT:    }
// CHECK-NEXT:  }
