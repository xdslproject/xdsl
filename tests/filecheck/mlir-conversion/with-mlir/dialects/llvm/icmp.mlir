// RUN: MLIR_GENERIC_ROUNDTRIP
// RUN: MLIR_ROUNDTRIP

%0, %1 = "test.op"() : () -> (i32, i32)

%eq = llvm.icmp "eq" %0, %1 : i32
// CHECK:  %{{\d+}} = llvm.icmp "eq" %0, %1 : i32

%ne = llvm.icmp "ne" %0, %1 : i32
// CHECK-NEXT:  %{{\d+}} = llvm.icmp "ne" %0, %1 : i32

%slt = llvm.icmp "slt" %0, %1 : i32
// CHECK-NEXT:  %{{\d+}} = llvm.icmp "slt" %0, %1 : i32

%sle = llvm.icmp "sle" %0, %1 : i32
// CHECK-NEXT:  %{{\d+}} = llvm.icmp "sle" %0, %1 : i32

%sgt = llvm.icmp "sgt" %0, %1 : i32
// CHECK-NEXT:  %{{\d+}} = llvm.icmp "sgt" %0, %1 : i32

%sge = llvm.icmp "sge" %0, %1 : i32
// CHECK-NEXT:  %{{\d+}} = llvm.icmp "sge" %0, %1 : i32

%ult = llvm.icmp "ult" %0, %1 : i32
// CHECK-NEXT:  %{{\d+}} = llvm.icmp "ult" %0, %1 : i32

%ule = llvm.icmp "ule" %0, %1 : i32
// CHECK-NEXT:  %{{\d+}} = llvm.icmp "ule" %0, %1 : i32

%ugt = llvm.icmp "ugt" %0, %1 : i32
// CHECK-NEXT:  %{{\d+}} = llvm.icmp "ugt" %0, %1 : i32

%uge = llvm.icmp "uge" %0, %1 : i32
// CHECK-NEXT:  %{{\d+}} = llvm.icmp "uge" %0, %1 : i32
