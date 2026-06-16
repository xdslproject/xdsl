// RUN: xdsl-opt -p x86-allocate-registers %s | filecheck %s


// CHECK-LABEL:    @external
x86_func.func @external() -> ()

// CHECK-LABEL:    @main_avx2
x86_func.func @main_avx2() {

// CHECK-NEXT:      %ymm0, %ymm1, %ymm2, %ymm3 = "test.op"() : () -> (!x86.avx2reg<ymm0>, !x86.avx2reg<ymm1>, !x86.avx2reg<ymm2>, !x86.avx2reg<ymm3>)
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

// CHECK-LABEL:    @main_avx512
x86_func.func @main_avx512() {
// CHECK-NEXT:      %zmm0, %zmm1, %zmm2, %zmm3 = "test.op"() : () -> (!x86.avx512reg<zmm0>, !x86.avx512reg<zmm1>, !x86.avx512reg<zmm2>, !x86.avx512reg<zmm3>)
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

// CHECK-LABEL:  @loops
x86_func.func @loops() {

// CHECK-NEXT:      %start, %end, %step = "test.op"() : () -> (!x86.reg64<rcx>, !x86.reg64<rdx>, !x86.reg64<rax>)
  %start, %end, %step = "test.op"() : () -> (!x86.reg64, !x86.reg64, !x86.reg64)

// CHECK-NEXT:      %for_result = x86_scf.for %iv : !x86.reg64<rcx>  = %start to %end step %step iter_args(%iter_val = %step) -> (!x86.reg64<rax>) {
// CHECK-NEXT:        %moved_val = x86.ds.mov %iter_val : (!x86.reg64<rax>) -> !x86.reg64<rax>
// CHECK-NEXT:        x86_scf.yield %moved_val : !x86.reg64<rax>
// CHECK-NEXT:      }
  %for_result = x86_scf.for %iv : !x86.reg64 = %start to %end step %step iter_args(%iter_val = %step) -> (!x86.reg64) {
    %moved_val = x86.ds.mov %iter_val : (!x86.reg64) -> !x86.reg64
    x86_scf.yield %moved_val : !x86.reg64
  }

// CHECK-NEXT:      x86_func.ret
  x86_func.ret
}
