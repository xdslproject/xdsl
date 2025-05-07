// RUN: xdsl-opt %s -p canonicalize --split-input-file | filecheck %s


builtin.module {
// CHECK-NEXT: builtin.module {

%0 = arith.constant 512 : i16
%1 = "csl.zeros"() : () -> memref<512xf32>
%2 = "csl.get_mem_dsd"(%1, %0) : (memref<512xf32>, i16) -> !csl<dsd mem1d_dsd>

%int8 = arith.constant 3 : i8
%3 = "csl.set_dsd_stride"(%2, %int8) : (!csl<dsd mem1d_dsd>, i8) -> !csl<dsd mem1d_dsd>

%4 = arith.constant 1 : i16
%5 = "csl.increment_dsd_offset"(%3, %4) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, i16) -> !csl<dsd mem1d_dsd>

%6 = arith.constant 510 : i16
%7 = "csl.set_dsd_length"(%5, %6) : (!csl<dsd mem1d_dsd>, i16) -> !csl<dsd mem1d_dsd>

"test.op"(%7) : (!csl<dsd mem1d_dsd>) -> ()

// CHECK-NEXT:  %0 = "csl.zeros"() : () -> memref<512xf32>
// CHECK-NEXT:  %1 = arith.constant 510 : i16
// CHECK-NEXT:  %2 = "csl.get_mem_dsd"(%0, %1) <{tensor_access = affine_map<(d0) -> (((d0 * 3) + 1))>}> : (memref<512xf32>, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:  "test.op"(%2) : (!csl<dsd mem1d_dsd>) -> ()



%8 = "test.op"() : () -> (!csl<dsd mem1d_dsd>)
%9 = arith.constant 2 : i16
%10 = "csl.increment_dsd_offset"(%8, %9) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, i16) -> !csl<dsd mem1d_dsd>
%11 = "csl.increment_dsd_offset"(%10, %9) <{"elem_type" = f32}> : (!csl<dsd mem1d_dsd>, i16) -> !csl<dsd mem1d_dsd>
%12 = arith.constant 509 : i16
%13 = arith.constant 511 : i16
%14 = "csl.set_dsd_length"(%11, %12) : (!csl<dsd mem1d_dsd>, i16) -> !csl<dsd mem1d_dsd>
%15 = "csl.set_dsd_length"(%14, %13) : (!csl<dsd mem1d_dsd>, i16) -> !csl<dsd mem1d_dsd>
%16 = arith.constant 2 : i8
%17 = arith.constant 3 : i8
%18 = "csl.set_dsd_stride"(%15, %16) : (!csl<dsd mem1d_dsd>, i8) -> !csl<dsd mem1d_dsd>
%19 = "csl.set_dsd_stride"(%18, %17) : (!csl<dsd mem1d_dsd>, i8) -> !csl<dsd mem1d_dsd>
"test.op"(%19) : (!csl<dsd mem1d_dsd>) -> ()

// CHECK-NEXT:  %3 = "test.op"() : () -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:  %4 = arith.constant 4 : i16
// CHECK-NEXT:  %5 = "csl.increment_dsd_offset"(%3, %4) <{elem_type = f32}> : (!csl<dsd mem1d_dsd>, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:  %6 = arith.constant 511 : i16
// CHECK-NEXT:  %7 = "csl.set_dsd_length"(%5, %6) : (!csl<dsd mem1d_dsd>, i16) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:  %8 = arith.constant 3 : i8
// CHECK-NEXT:  %9 = "csl.set_dsd_stride"(%7, %8) : (!csl<dsd mem1d_dsd>, i8) -> !csl<dsd mem1d_dsd>
// CHECK-NEXT:  "test.op"(%9) : (!csl<dsd mem1d_dsd>) -> ()

}
// CHECK-NEXT: }
