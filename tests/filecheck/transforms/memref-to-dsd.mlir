// RUN: xdsl-opt %s -p memref-to-dsd | filecheck %s

builtin.module {
// CHECK-NEXT: builtin.module {

  %0 = "test.op"() : () -> (index)
  "memref.global"() <{"sym_name" = "a", "type" = memref<512xf32>, "initial_value", "sym_visibility" = "public"}> : () -> ()
  "memref.global"() <{"sym_name" = "b", "type" = memref<510xf32>, "initial_value", "sym_visibility" = "public"}> : () -> ()
  %a = memref.get_global @a : memref<512xf32>
  %b = memref.get_global @b : memref<510xf32>
  %c = memref.get_global @b : memref<1024xf32>
  %11 = memref.subview %a[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
  %12 = memref.subview %a[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
  "csl.fadds"(%b, %11, %12) : (memref<510xf32>, memref<510xf32, strided<[1]>>, memref<510xf32, strided<[1], offset: 2>>) -> ()
  %13 = memref.subview %c[1] [510] [2] : memref<1024xf32> to memref<510xf32, strided<[2], offset: 1>>
  "csl.fadds"(%b, %b, %13) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[2], offset: 1>>) -> ()

// CHECK-NEXT: %0 = "test.op"() : () -> index
// CHECK-NEXT: "memref.global"() <{"sym_name" = "a", "type" = memref<512xf32>, "initial_value", "sym_visibility" = "public"}> : () -> ()
// CHECK-NEXT: "memref.global"() <{"sym_name" = "b", "type" = memref<510xf32>, "initial_value", "sym_visibility" = "public"}> : () -> ()
// CHECK-NEXT: %a = "csl.get_mem_dsd"() <{"sym_name" = @a, "sizes" = [512 : i16], "operandSegmentSizes" = array<i32: 0, 0>}> : () -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %b = "csl.get_mem_dsd"() <{"sym_name" = @b, "sizes" = [510 : i16], "operandSegmentSizes" = array<i32: 0, 0>}> : () -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %c = "csl.get_mem_dsd"() <{"sym_name" = @b, "sizes" = [1024 : i16], "operandSegmentSizes" = array<i32: 0, 0>}> : () -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %1 = arith.constant 510 : ui16
// CHECK-NEXT: %2 = "csl.set_dsd_length"(%a, %1) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %3 = arith.constant 510 : ui16
// CHECK-NEXT: %4 = "csl.set_dsd_length"(%a, %3) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: "csl.fadds"(%b, %2, %4) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT: %5 = arith.constant 510 : ui16
// CHECK-NEXT: %6 = "csl.set_dsd_length"(%c, %5) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %7 = arith.constant 2 : si8
// CHECK-NEXT: %8 = "csl.set_dsd_stride"(%6, %7) : (!csl<dsd mem1d_dsd>, si8) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: "csl.fadds"(%b, %b, %8) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()


  "memref.global"() <{"sym_name" = "int_buf1", "type" = memref<10xi32>, "initial_value", "sym_visibility" = "public"}> : () -> ()
  "memref.global"() <{"sym_name" = "int_buf2", "type" = memref<10xi32>, "initial_value", "sym_visibility" = "public"}> : () -> ()
  %23 = memref.get_global @int_buf1 : memref<10xi32>
  %24 = memref.get_global @int_buf2 : memref<10xi32>
  "memref.copy"(%23, %24) : (memref<10xi32>, memref<10xi32>) -> ()

// CHECK-NEXT:      "memref.global"() <{"sym_name" = "int_buf1", "type" = memref<10xi32>, "initial_value"}> : () -> ()
// CHECK-NEXT: "memref.global"() <{"sym_name" = "int_buf2", "type" = memref<10xi32>, "initial_value"}> : () -> ()
// CHECK-NEXT: %9 = "csl.get_mem_dsd"() <{"sym_name" = @int_buf1, "sizes" = [10 : i16], "operandSegmentSizes" = array<i32: 0, 0>}> : () -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %10 = "csl.get_mem_dsd"() <{"sym_name" = @int_buf2, "sizes" = [10 : i16], "operandSegmentSizes" = array<i32: 0, 0>}> : () -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: "csl.mov32"(%10, %9) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()


}
// CHECK-NEXT: }
