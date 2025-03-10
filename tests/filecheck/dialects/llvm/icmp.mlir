// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | filecheck %s

%arg0, %arg1 = "test.op"() : () -> (i32, i32)

%icmp_eq_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 0 : i64}> : (i32, i32) -> i1
%icmp_eq = llvm.icmp "eq" %arg0, %arg1 : i32
// CHECK:  %icmp_eq_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 0 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %icmp_eq = "llvm.icmp"(%arg0, %arg1) <{predicate = 0 : i64}> : (i32, i32) -> i1

%icmp_ne_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 1 : i64}> : (i32, i32) -> i1
%icmp_ne = llvm.icmp "ne" %arg0, %arg1 : i32
// CHECK-NEXT:  %icmp_ne_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 1 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %icmp_ne = "llvm.icmp"(%arg0, %arg1) <{predicate = 1 : i64}> : (i32, i32) -> i1

%icmp_slt_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 2 : i64}> : (i32, i32) -> i1
%icmp_slt = llvm.icmp "slt" %arg0, %arg1 : i32
// CHECK-NEXT:  %icmp_slt_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 2 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %icmp_slt = "llvm.icmp"(%arg0, %arg1) <{predicate = 2 : i64}> : (i32, i32) -> i1

%icmp_sle_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 3 : i64}> : (i32, i32) -> i1
%icmp_sle = llvm.icmp "sle" %arg0, %arg1 : i32
// CHECK-NEXT:  %icmp_sle_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 3 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %icmp_sle = "llvm.icmp"(%arg0, %arg1) <{predicate = 3 : i64}> : (i32, i32) -> i1

%icmp_sgt_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 4 : i64}> : (i32, i32) -> i1
%icmp_sgt = llvm.icmp "sgt" %arg0, %arg1 : i32
// CHECK-NEXT:  %icmp_sgt_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 4 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %icmp_sgt = "llvm.icmp"(%arg0, %arg1) <{predicate = 4 : i64}> : (i32, i32) -> i1

%icmp_sge_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 5 : i64}> : (i32, i32) -> i1
%icmp_sge = llvm.icmp "sge" %arg0, %arg1 : i32
// CHECK-NEXT:  %icmp_sge_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 5 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %icmp_sge = "llvm.icmp"(%arg0, %arg1) <{predicate = 5 : i64}> : (i32, i32) -> i1

%icmp_ult_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 6 : i64}> : (i32, i32) -> i1
%icmp_ult = llvm.icmp "ult" %arg0, %arg1 : i32
// CHECK-NEXT:  %icmp_ult_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 6 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %icmp_ult = "llvm.icmp"(%arg0, %arg1) <{predicate = 6 : i64}> : (i32, i32) -> i1

%icmp_ule_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 7 : i64}> : (i32, i32) -> i1
%icmp_ule = llvm.icmp "ule" %arg0, %arg1 : i32
// CHECK-NEXT:  %icmp_ule_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 7 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %icmp_ule = "llvm.icmp"(%arg0, %arg1) <{predicate = 7 : i64}> : (i32, i32) -> i1

%icmp_ugt_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 8 : i64}> : (i32, i32) -> i1
%icmp_ugt = llvm.icmp "ugt" %arg0, %arg1 : i32
// CHECK-NEXT:  %icmp_ugt_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 8 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %icmp_ugt = "llvm.icmp"(%arg0, %arg1) <{predicate = 8 : i64}> : (i32, i32) -> i1

%icmp_uge_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 9 : i64}> : (i32, i32) -> i1
%icmp_uge = llvm.icmp "uge" %arg0, %arg1 : i32
// CHECK-NEXT:  %icmp_uge_p = "llvm.icmp"(%arg0, %arg1) <{predicate = 9 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:  %icmp_uge = "llvm.icmp"(%arg0, %arg1) <{predicate = 9 : i64}> : (i32, i32) -> i1
