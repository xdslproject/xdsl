// RUN: xdsl-opt %s -p memref-to-dsd | filecheck %s

builtin.module {
"csl.module"() <{"kind" = #csl<module_kind program>}> ({
// CHECK-NEXT: builtin.module {
// CHECK-NEXT: "csl.module"() <{"kind" = #csl<module_kind program>}> ({

  %0 = "test.op"() : () -> (index)
  %a = memref.alloc() {"alignment" = 64 : i64} : memref<512xf32>
  %b = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
  %c = memref.alloc() {"alignment" = 64 : i64} : memref<1024xf32>
  %d = memref.subview %a[0] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1]>>
  %e = memref.subview %a[2] [510] [1] : memref<512xf32> to memref<510xf32, strided<[1], offset: 2>>
  "csl.fadds"(%b, %d, %e) : (memref<510xf32>, memref<510xf32, strided<[1]>>, memref<510xf32, strided<[1], offset: 2>>) -> ()
  %f = memref.subview %c[1] [510] [2] : memref<1024xf32> to memref<510xf32, strided<[2], offset: 1>>
  "csl.fadds"(%b, %b, %f) : (memref<510xf32>, memref<510xf32>, memref<510xf32, strided<[2], offset: 1>>) -> ()

  %1 = "csl.addressof"(%a) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
  %2 = "csl.addressof"(%b) : (memref<510xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
  %3 = "csl.addressof"(%c) : (memref<1024xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
  "csl.export"(%1) <{"var_name" = "a", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
  "csl.export"(%2) <{"var_name" = "b", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
  "csl.export"(%2) <{"var_name" = "c", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()

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
// CHECK-NEXT: %e_2 = arith.constant 2 : si16
// CHECK-NEXT: %e_3 = "csl.increment_dsd_offset"(%e_1, %e_2) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, si16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: "csl.fadds"(%b_2, %d_1, %e_3) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT: %f = arith.constant 510 : ui16
// CHECK-NEXT: %f_1 = "csl.set_dsd_length"(%c_2, %f) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %f_2 = arith.constant 2 : si8
// CHECK-NEXT: %f_3 = "csl.set_dsd_stride"(%f_1, %f_2) : (!csl<dsd mem1d_dsd>, si8) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %f_4 = arith.constant 1 : si16
// CHECK-NEXT: %f_5 = "csl.increment_dsd_offset"(%f_3, %f_4) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, si16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: "csl.fadds"(%b_2, %b_2, %f_5) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()
// CHECK-NEXT: %1 = "csl.addressof"(%a) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT: %2 = "csl.addressof"(%b) : (memref<510xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT: %3 = "csl.addressof"(%c) : (memref<1024xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT: "csl.export"(%1) <{"var_name" = "a", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT: "csl.export"(%2) <{"var_name" = "b", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT: "csl.export"(%2) <{"var_name" = "c", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()


  %23 = memref.alloc() {"alignment" = 64 : i64} : memref<10xi32>
  %24 = memref.alloc() {"alignment" = 64 : i64} : memref<10xi32>
  "memref.copy"(%23, %24) : (memref<10xi32>, memref<10xi32>) -> ()

// CHECK:      %4 = "csl.zeros"() : () -> memref<10xi32>
// CHECK-NEXT: %5 = arith.constant 10 : i16
// CHECK-NEXT: %6 = "csl.get_mem_dsd"(%4, %5) : (memref<10xi32>, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %7 = "csl.zeros"() : () -> memref<10xi32>
// CHECK-NEXT: %8 = arith.constant 10 : i16
// CHECK-NEXT: %9 = "csl.get_mem_dsd"(%7, %8) : (memref<10xi32>, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: "csl.mov32"(%9, %6) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()


%25 = arith.constant 0 : index
%26 = arith.constant 510 : index
%27 = arith.constant 1 : index
%28 = arith.constant 2 : index
%29 = memref.subview %b[%25] [%26] [%27] : memref<510xf32> to memref<510xf32, strided<[1]>>
%30 = memref.subview %c[%27] [%26] [%28] : memref<1024xf32> to memref<510xf32, strided<[2], offset: 1>>

// CHECK-NEXT: %10 = arith.constant 0 : index
// CHECK-NEXT: %11 = arith.constant 510 : index
// CHECK-NEXT: %12 = arith.constant 1 : index
// CHECK-NEXT: %13 = arith.constant 2 : index
// CHECK-NEXT: %14 = arith.index_cast %11 : index to ui16
// CHECK-NEXT: %15 = "csl.set_dsd_length"(%b_2, %14) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %16 = arith.index_cast %12 : index to si8
// CHECK-NEXT: %17 = "csl.set_dsd_stride"(%15, %16) : (!csl<dsd mem1d_dsd>, si8) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %18 = arith.index_cast %10 : index to si16
// CHECK-NEXT: %19 = "csl.increment_dsd_offset"(%17, %18) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, si16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %20 = arith.index_cast %11 : index to ui16
// CHECK-NEXT: %21 = "csl.set_dsd_length"(%c_2, %20) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %22 = arith.index_cast %13 : index to si8
// CHECK-NEXT: %23 = "csl.set_dsd_stride"(%21, %22) : (!csl<dsd mem1d_dsd>, si8) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: %24 = arith.index_cast %12 : index to si16
// CHECK-NEXT: %25 = "csl.increment_dsd_offset"(%23, %24) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, si16) -> !csl<dsd mem1d_dsd>

%31 = "test.op"() : () -> (!csl<dsd mem1d_dsd>)
%32 = builtin.unrealized_conversion_cast %31 : !csl<dsd mem1d_dsd> to memref<255xf32>
"csl.fadds"(%32, %32, %32) : (memref<255xf32>, memref<255xf32>, memref<255xf32>) -> ()

// CHECK-NEXT: %26 = "test.op"() : () -> !csl<dsd mem1d_dsd>
// CHECK-NEXT: "csl.fadds"(%26, %26, %26) : (!csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>, !csl<dsd mem1d_dsd>) -> ()

}) {sym_name = "program"} :  () -> ()
}
// CHECK-NEXT: }) {"sym_name" = "program"} :  () -> ()
// CHECK-NEXT: }
