// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

builtin.module {
  %0 = arith.constant 0 : i64
  %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
  %3 = llvm.mlir.zero : !llvm.ptr
  %zero_struct = llvm.mlir.zero : !llvm.struct<(i32, f32)>
  %zero_addrspace = llvm.mlir.zero : !llvm.ptr<1>
  %4 = llvm.alloca %0 x i64 {alignment = 32 : i64} : (i64) -> !llvm.ptr
  %6 = llvm.alloca %0 x i32 {alignment = 32 : i64} : (i64) -> !llvm.ptr
  %7 = llvm.getelementptr %6[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  llvm.store volatile %5, %6 {alignment = 32 : i64, nontemporal} : i64, !llvm.ptr
  %2 = llvm.load %1 : !llvm.ptr -> i32
  %5 = llvm.load %4 : !llvm.ptr -> i64
  %8 = llvm.load %4 {alignment = 16 : i64} : !llvm.ptr -> i32
  %9 = llvm.load %4 atomic unordered {alignment = 32 : i64} : !llvm.ptr -> i32
  %lm = llvm.load %4 atomic monotonic {alignment = 4 : i64} : !llvm.ptr -> i32
  %la = llvm.load %4 atomic acquire {alignment = 4 : i64} : !llvm.ptr -> i32
  %lr = llvm.load %4 atomic seq_cst {alignment = 4 : i64} : !llvm.ptr -> i32
  %ptr_int = llvm.ptrtoint %1 : !llvm.ptr to i64

  %agg = "test.op"() : () -> !llvm.struct<(i32, !llvm.array<3 x i32>)>
  %ev_field = llvm.extractvalue %agg[0] : !llvm.struct<(i32, !llvm.array<3 x i32>)>
  %ev_arr = llvm.extractvalue %agg[1, 2] : !llvm.struct<(i32, !llvm.array<3 x i32>)>
  %iv = llvm.insertvalue %ev_field, %agg[0] : !llvm.struct<(i32, !llvm.array<3 x i32>)>
  %iv_arr = llvm.insertvalue %ev_field, %agg[1, 0] : !llvm.struct<(i32, !llvm.array<3 x i32>)>
}

// CHECK:       builtin.module {
// CHECK-NEXT:    [[CST:%.*]] = arith.constant 0 : i64
// CHECK-NEXT:    [[PTR:%.*]] = llvm.inttoptr [[CST]] : i64 to !llvm.ptr
// CHECK-NEXT:    {{%.*}} = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    {{%.*}} = llvm.mlir.zero : !llvm.struct<(i32, f32)>
// CHECK-NEXT:    {{%.*}} = llvm.mlir.zero : !llvm.ptr<1>
// CHECK-NEXT:    [[ALLOCA64:%.*]] = llvm.alloca [[CST]] x i64 {alignment = 32 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    [[ALLOCA32:%.*]] = llvm.alloca [[CST]] x i32 {alignment = 32 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    {{%.*}} = llvm.getelementptr [[ALLOCA32]][[[CST]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store volatile [[STORED:%.*]], [[ALLOCA32]] {alignment = 32 : i64, nontemporal} : i64, !llvm.ptr
// CHECK-NEXT:    {{%.*}} = llvm.load [[PTR]] : !llvm.ptr -> i32
// CHECK-NEXT:    [[STORED]] = llvm.load [[ALLOCA64]] : !llvm.ptr -> i64
// CHECK-NEXT:    {{%.*}} = llvm.load [[ALLOCA64]] {alignment = 16 : i64} : !llvm.ptr -> i32
// CHECK-NEXT:    {{%.*}} = llvm.load [[ALLOCA64]] atomic unordered {alignment = 32 : i64} : !llvm.ptr -> i32
// CHECK-NEXT:    {{%.*}} = llvm.load [[ALLOCA64]] atomic monotonic {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK-NEXT:    {{%.*}} = llvm.load [[ALLOCA64]] atomic acquire {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK-NEXT:    {{%.*}} = llvm.load [[ALLOCA64]] atomic seq_cst {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK-NEXT:    {{%.*}} = llvm.ptrtoint [[PTR]] : !llvm.ptr to i64
// CHECK-NEXT:    [[AGG:%.*]] = "test.op"() : () -> !llvm.struct<(i32, !llvm.array<3 x i32>)>
// CHECK-NEXT:    [[EVF:%.*]] = llvm.extractvalue [[AGG]][0] : !llvm.struct<(i32, !llvm.array<3 x i32>)>
// CHECK-NEXT:    {{%.*}} = llvm.extractvalue [[AGG]][1, 2] : !llvm.struct<(i32, !llvm.array<3 x i32>)>
// CHECK-NEXT:    {{%.*}} = llvm.insertvalue [[EVF]], [[AGG]][0] : !llvm.struct<(i32, !llvm.array<3 x i32>)>
// CHECK-NEXT:    {{%.*}} = llvm.insertvalue [[EVF]], [[AGG]][1, 0] : !llvm.struct<(i32, !llvm.array<3 x i32>)>
// CHECK-NEXT:  }
