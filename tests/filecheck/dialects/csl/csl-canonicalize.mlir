// RUN: xdsl-opt %s -p canonicalize --split-input-file | filecheck %s


builtin.module {
// CHECK-NEXT: builtin.module {

%0 = arith.constant 512 : i16
%1 = "csl.zeros"() : () -> memref<512xf32>
%2 = "csl.get_mem_dsd"(%1, %0) : (memref<512xf32>, i16) -> !csl<dsd mem1d_dsd>

%3 = arith.constant 1 : si16
%4 = "csl.increment_dsd_offset"(%2, %3) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, si16) -> !csl<dsd mem1d_dsd>

%5 = arith.constant 510 : ui16
%6 = "csl.set_dsd_length"(%4, %5) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>

%int8 = arith.constant 1 : si8
%7 = "csl.set_dsd_stride"(%6, %int8) : (!csl<dsd mem1d_dsd>, si8) -> !csl<dsd mem1d_dsd>

"test.op"(%7) : (!csl<dsd mem1d_dsd>) -> ()

// CHECK-NEXT:  %0 = "csl.zeros"() : () -> memref<512xf32>
// CHECK-NEXT:  %1 = arith.constant 510 : ui16
// CHECK-NEXT:  %2 = "csl.get_mem_dsd"(%0, %1) <{"offsets" = [1 : si16], "strides" = [1 : si8]}> : (memref<512xf32>, ui16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:  "test.op"(%2) : (!csl<dsd mem1d_dsd>) -> ()



%8 = "test.op"() : () -> (!csl<dsd mem1d_dsd>)
%9 = arith.constant 2 : si16
%10 = "csl.increment_dsd_offset"(%8, %9) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, si16) -> !csl<dsd mem1d_dsd>
%11 = "csl.increment_dsd_offset"(%10, %9) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, si16) -> !csl<dsd mem1d_dsd>
%12 = arith.constant 509 : ui16
%13 = arith.constant 511 : ui16
%14 = "csl.set_dsd_length"(%11, %12) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
%15 = "csl.set_dsd_length"(%14, %13) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
%16 = arith.constant 2 : si8
%17 = arith.constant 3 : si8
%18 = "csl.set_dsd_stride"(%15, %16) : (!csl<dsd mem1d_dsd>, si8) -> !csl<dsd mem1d_dsd>
%19 = "csl.set_dsd_stride"(%18, %17) : (!csl<dsd mem1d_dsd>, si8) -> !csl<dsd mem1d_dsd>
"test.op"(%19) : (!csl<dsd mem1d_dsd>) -> ()

// CHECK-NEXT:  %3 = "test.op"() : () -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:  %4 = arith.constant 2 : si16
// CHECK-NEXT:  %5 = arith.addi %4, %4 : si16
// CHECK-NEXT:  %6 = "csl.increment_dsd_offset"(%3, %5) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, si16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:  %7 = arith.constant 511 : ui16
// CHECK-NEXT:  %8 = "csl.set_dsd_length"(%6, %7) : (!csl<dsd mem1d_dsd>, ui16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:  %9 = arith.constant 3 : si8
// CHECK-NEXT:  %10 = "csl.set_dsd_stride"(%8, %9) : (!csl<dsd mem1d_dsd>, si8) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:  "test.op"(%10) : (!csl<dsd mem1d_dsd>) -> ()

}
// CHECK-NEXT: }
