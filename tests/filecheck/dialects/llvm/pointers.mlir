// RUN: XDSL_ROUNDTRIP
builtin.module {
  %0 = arith.constant 0 : i64
  %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
  %3 = llvm.mlir.zero : !llvm.ptr
  %4 = llvm.alloca %0 x index {alignment = 32 : i64} : (i64) -> !llvm.ptr
  %6 = llvm.getelementptr %4[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  %7 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
  %ib = llvm.getelementptr inbounds %4[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  %gep_mixed = llvm.getelementptr %4[1, %0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x !llvm.struct<(i32, i32, i32)>>
  %2 = llvm.load %1 : !llvm.ptr -> i32
  %5 = llvm.load %4 : !llvm.ptr -> index
  %8 = llvm.load %4 {alignment = 16 : i64} : !llvm.ptr -> index
  %9 = llvm.load %4 atomic unordered {alignment = 32 : i64} : !llvm.ptr -> index
  %lm = llvm.load %4 atomic monotonic : !llvm.ptr -> index
  %la = llvm.load %4 atomic acquire : !llvm.ptr -> index
  %lr = llvm.load %4 atomic release : !llvm.ptr -> index
  %lar = llvm.load %4 atomic acq_rel : !llvm.ptr -> index
  %lsc = llvm.load %4 atomic seq_cst : !llvm.ptr -> index

  %v = arith.constant 0 : i32
  llvm.store %v, %4 : i32, !llvm.ptr
  llvm.store %v, %4 {alignment = 8 : i64} : i32, !llvm.ptr
  llvm.store volatile %v, %4 : i32, !llvm.ptr
  llvm.store %v, %4 {nontemporal} : i32, !llvm.ptr
  llvm.store volatile %v, %4 {alignment = 16 : i64, nontemporal} : i32, !llvm.ptr

  %agg = "test.op"() : () -> !llvm.struct<(i32, !llvm.array<3 x i32>)>
  %ev_field = llvm.extractvalue %agg[0] : !llvm.struct<(i32, !llvm.array<3 x i32>)>
  %ev_arr = llvm.extractvalue %agg[1, 2] : !llvm.struct<(i32, !llvm.array<3 x i32>)>
  %iv = llvm.insertvalue %v, %agg[0] : !llvm.struct<(i32, !llvm.array<3 x i32>)>
  %iv_arr = llvm.insertvalue %v, %agg[1, 0] : !llvm.struct<(i32, !llvm.array<3 x i32>)>
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = arith.constant 0 : i64
// CHECK-NEXT:    %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
// CHECK-NEXT:    %2 = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %3 = llvm.alloca %0 x index {alignment = 32 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %4 = llvm.getelementptr %3[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-NEXT:    %5 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %ib = llvm.getelementptr inbounds %3[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-NEXT:    %gep_mixed = llvm.getelementptr %3[1, %0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x !llvm.struct<(i32, i32, i32)>>
// CHECK-NEXT:    %6 = llvm.load %1 : !llvm.ptr -> i32
// CHECK-NEXT:    %7 = llvm.load %3 : !llvm.ptr -> index
// CHECK-NEXT:    %8 = llvm.load %3 {alignment = 16 : i64} : !llvm.ptr -> index
// CHECK-NEXT:    %9 = llvm.load %3 atomic unordered {alignment = 32 : i64} : !llvm.ptr -> index
// CHECK-NEXT:    %lm = llvm.load %3 atomic monotonic : !llvm.ptr -> index
// CHECK-NEXT:    %la = llvm.load %3 atomic acquire : !llvm.ptr -> index
// CHECK-NEXT:    %lr = llvm.load %3 atomic release : !llvm.ptr -> index
// CHECK-NEXT:    %lar = llvm.load %3 atomic acq_rel : !llvm.ptr -> index
// CHECK-NEXT:    %lsc = llvm.load %3 atomic seq_cst : !llvm.ptr -> index
// CHECK-NEXT:    %v = arith.constant 0 : i32
// CHECK-NEXT:    llvm.store %v, %3 : i32, !llvm.ptr
// CHECK-NEXT:    llvm.store %v, %3 {alignment = 8 : i64} : i32, !llvm.ptr
// CHECK-NEXT:    llvm.store volatile %v, %3 : i32, !llvm.ptr
// CHECK-NEXT:    llvm.store %v, %3 {nontemporal} : i32, !llvm.ptr
// CHECK-NEXT:    llvm.store volatile %v, %3 {alignment = 16 : i64, nontemporal} : i32, !llvm.ptr
// CHECK-NEXT:    %agg = "test.op"() : () -> !llvm.struct<(i32, !llvm.array<3 x i32>)>
// CHECK-NEXT:    %ev_field = llvm.extractvalue %agg[0] : !llvm.struct<(i32, !llvm.array<3 x i32>)>
// CHECK-NEXT:    %ev_arr = llvm.extractvalue %agg[1, 2] : !llvm.struct<(i32, !llvm.array<3 x i32>)>
// CHECK-NEXT:    %iv = llvm.insertvalue %v, %agg[0] : !llvm.struct<(i32, !llvm.array<3 x i32>)>
// CHECK-NEXT:    %iv_arr = llvm.insertvalue %v, %agg[1, 0] : !llvm.struct<(i32, !llvm.array<3 x i32>)>
// CHECK-NEXT:  }
