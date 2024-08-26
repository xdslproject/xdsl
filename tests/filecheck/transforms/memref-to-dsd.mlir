// RUN: xdsl-opt %s -p memref-to-dsd | filecheck %s

builtin.module {
// CHECK-NEXT: builtin.module {

  %0 = "test.op"() : () -> (index)
  %a = memref.alloc() {"alignment" = 64 : i64} : memref<512xf32>
  %b = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
  %c = memref.alloc() {"alignment" = 64 : i64} : memref<1024xf32>
  %d = memref.subview %a[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
  %e = memref.subview %a[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
  "csl.fadds"(%b, %d, %e) : (memref<510xf32>, memref<510xf32, strided<[1]>>, memref<510xf32, strided<[1], offset: 2>>) -> ()
  %f = memref.subview %c[1] [510] [2] : memref<1024xf32> to memref<510xf32, strided<[2], offset: 1>>
  "csl.fadds"(%b, %b, %f) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[2], offset: 1>>) -> ()

// CHECK-NEXT: %0 = "test.op"() : () -> index
// CHECK-NEXT: %a = "csl.zeros"() : () -> memref<512xf32>
// CHECK-NEXT: %a_1 = arith.constant 512 : i16
// CHECK-NEXT: %a_2 = "csl.get_mem_dsd"(%a, %a_1) : (memref<512xf32>, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %b = "csl.zeros"() : () -> memref<510xf32>
// CHECK-NEXT: %b_1 = arith.constant 510 : i16
// CHECK-NEXT: %b_2 = "csl.get_mem_dsd"(%b, %b_1) : (memref<510xf32>, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %c = "csl.zeros"() : () -> memref<1024xf32>
// CHECK-NEXT: %c_1 = arith.constant 1024 : i16
// CHECK-NEXT: %c_2 = "csl.get_mem_dsd"(%c, %c_1) : (memref<1024xf32>, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %d = arith.constant 510 : ui16
// CHECK-NEXT: %d_1 = "csl.set_dsd_length"(%a_2, %d) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %e = arith.constant 510 : ui16
// CHECK-NEXT: %e_1 = "csl.set_dsd_length"(%a_2, %e) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: "csl.fadds"(%b_2, %d_1, %e_1) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT: %f = arith.constant 510 : ui16
// CHECK-NEXT: %f_1 = "csl.set_dsd_length"(%c_2, %f) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %f_2 = arith.constant 2 : si8
// CHECK-NEXT: %f_3 = "csl.set_dsd_stride"(%f_1, %f_2) : (!csl<dsd mem1d_dsd>, si8) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: "csl.fadds"(%b_2, %b_2, %f_3) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()


  %23 = memref.alloc() {"alignment" = 64 : i64} : memref<10xi32>
  %24 = memref.alloc() {"alignment" = 64 : i64} : memref<10xi32>
  "memref.copy"(%23, %24) : (memref<10xi32>, memref<10xi32>) -> ()

// CHECK:      %1 = "csl.zeros"() : () -> memref<10xi32>
// CHECK-NEXT: %2 = arith.constant 10 : i16
// CHECK-NEXT: %3 = "csl.get_mem_dsd"(%1, %2) : (memref<10xi32>, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %4 = "csl.zeros"() : () -> memref<10xi32>
// CHECK-NEXT: %5 = arith.constant 10 : i16
// CHECK-NEXT: %6 = "csl.get_mem_dsd"(%4, %5) : (memref<10xi32>, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: "csl.mov32"(%6, %3) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()


}
// CHECK-NEXT: }
