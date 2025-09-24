// RUN: xdsl-opt -p x86-allocate-registers %s | filecheck %s

// CHECK:       builtin.module {

// CHECK-NEXT:    x86_func.func @external() -> ()
x86_func.func @external() -> ()


// CHECK-NEXT:    x86_func.func @main_avx2() {
// CHECK-NEXT:      %ymm0, %ymm1, %ymm2, %ymm3 = "test.op"() : () -> (!x86.avx2reg<ymm0>, !x86.avx2reg<ymm1>, !x86.avx2reg<ymm2>, !x86.avx2reg<ymm3>)
x86_func.func @main_avx2() {
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

// CHECK-NEXT:    x86_func.func @main_avx512() {
// CHECK-NEXT:      %zmm0, %zmm1, %zmm2, %zmm3 = "test.op"() : () -> (!x86.avx512reg<zmm0>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm3>)
x86_func.func @main_avx512() {
  %zmm0, %zmm1, %zmm2, %zmm3 = "test.op"() : () -> (!x86.avx512reg<zmm0>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm3>)

// inout is allocated to same register
// CHECK-NEXT:      %r0 = x86.rss.vfmadd231pd %zmm0, %zmm1, %zmm2 : (!x86.avx512reg<zmm0>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm2>) -> !x86.avx512reg<zmm0>
  %r0 = x86.rss.vfmadd231pd %zmm0, %zmm1, %zmm2 : (!x86.avx512reg<zmm0>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm2>) -> !x86.avx512reg

// read-only is allocated to new register
// CHECK-NEXT:      %u0 = "test.op"() : () -> !x86.avx512reg<zmm4>
// CHECK-NEXT:      %r1 = x86.rss.vfmadd231pd %zmm3, %u0, %zmm2 : (!x86.avx512reg<zmm3>, !x86.avx512reg<zmm4>, !x86.avx512reg<zmm2>) -> !x86.avx512reg<zmm3>
  %u0 = "test.op"() : () -> !x86.avx512reg
  %r1 = x86.rss.vfmadd231pd %zmm3, %u0, %zmm2 : (!x86.avx512reg<zmm3>, !x86.avx512reg, !x86.avx512reg<zmm2>) -> !x86.avx512reg

// CHECK-NEXT:      x86_func.ret
  x86_func.ret
}

// CHECK-NEXT:    }

// CHECK-NEXT:  }
