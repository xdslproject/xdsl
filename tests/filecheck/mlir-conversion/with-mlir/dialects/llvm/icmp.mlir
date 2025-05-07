// RUN: mlir-opt %s --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt --print-op-generic | filecheck %s

%0, %1 = "test.op"() : () -> (i32, i32)

%2 = "llvm.icmp"(%0, %1) <{predicate = 0 : i64}> : (i32, i32) -> i1
%3 = llvm.icmp "eq" %0, %1 : i32
// CHECK:  %2 = "llvm.icmp"(%0, %1) <{predicate = 0 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %3 = "llvm.icmp"(%0, %1) <{predicate = 0 : i64}> : (i32, i32) -> i1

%4 = "llvm.icmp"(%0, %1) <{predicate = 1 : i64}> : (i32, i32) -> i1
%5 = llvm.icmp "ne" %0, %1 : i32
// CHECK-NEXT:  %4 = "llvm.icmp"(%0, %1) <{predicate = 1 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %5 = "llvm.icmp"(%0, %1) <{predicate = 1 : i64}> : (i32, i32) -> i1

%6 = "llvm.icmp"(%0, %1) <{predicate = 2 : i64}> : (i32, i32) -> i1
%7 = llvm.icmp "slt" %0, %1 : i32
// CHECK-NEXT:  %6 = "llvm.icmp"(%0, %1) <{predicate = 2 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %7 = "llvm.icmp"(%0, %1) <{predicate = 2 : i64}> : (i32, i32) -> i1

%8 = "llvm.icmp"(%0, %1) <{predicate = 3 : i64}> : (i32, i32) -> i1
%9 = llvm.icmp "sle" %0, %1 : i32
// CHECK-NEXT:  %8 = "llvm.icmp"(%0, %1) <{predicate = 3 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %9 = "llvm.icmp"(%0, %1) <{predicate = 3 : i64}> : (i32, i32) -> i1

%10 = "llvm.icmp"(%0, %1) <{predicate = 4 : i64}> : (i32, i32) -> i1
%11 = llvm.icmp "sgt" %0, %1 : i32
// CHECK-NEXT:  %10 = "llvm.icmp"(%0, %1) <{predicate = 4 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %11 = "llvm.icmp"(%0, %1) <{predicate = 4 : i64}> : (i32, i32) -> i1

%12 = "llvm.icmp"(%0, %1) <{predicate = 5 : i64}> : (i32, i32) -> i1
%13 = llvm.icmp "sge" %0, %1 : i32
// CHECK-NEXT:  %12 = "llvm.icmp"(%0, %1) <{predicate = 5 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %13 = "llvm.icmp"(%0, %1) <{predicate = 5 : i64}> : (i32, i32) -> i1

%14 = "llvm.icmp"(%0, %1) <{predicate = 6 : i64}> : (i32, i32) -> i1
%15 = llvm.icmp "ult" %0, %1 : i32
// CHECK-NEXT:  %14 = "llvm.icmp"(%0, %1) <{predicate = 6 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %15 = "llvm.icmp"(%0, %1) <{predicate = 6 : i64}> : (i32, i32) -> i1

%16 = "llvm.icmp"(%0, %1) <{predicate = 7 : i64}> : (i32, i32) -> i1
%17 = llvm.icmp "ule" %0, %1 : i32
// CHECK-NEXT:  %16 = "llvm.icmp"(%0, %1) <{predicate = 7 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %17 = "llvm.icmp"(%0, %1) <{predicate = 7 : i64}> : (i32, i32) -> i1

%18 = "llvm.icmp"(%0, %1) <{predicate = 8 : i64}> : (i32, i32) -> i1
%19 = llvm.icmp "ugt" %0, %1 : i32
// CHECK-NEXT:  %18 = "llvm.icmp"(%0, %1) <{predicate = 8 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %19 = "llvm.icmp"(%0, %1) <{predicate = 8 : i64}> : (i32, i32) -> i1

%20 = "llvm.icmp"(%0, %1) <{predicate = 9 : i64}> : (i32, i32) -> i1
%21 = llvm.icmp "uge" %0, %1 : i32
// CHECK-NEXT:  %20 = "llvm.icmp"(%0, %1) <{predicate = 9 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %21 = "llvm.icmp"(%0, %1) <{predicate = 9 : i64}> : (i32, i32) -> i1
