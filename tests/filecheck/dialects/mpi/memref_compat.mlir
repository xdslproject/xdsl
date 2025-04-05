// RUN: xdsl-opt -p lower-mpi --print-op-generic %s
"builtin.module"() ({
    %ref = "memref.alloc"() {"alignment" = 32 : i64, operandSegmentSizes = array<i32: 0, 0>} : () -> memref<100x14x14xf64>
    %buff, %count, %dtype = "mpi.unwrap_memref"(%ref) : (memref<100x14x14xf64>) -> (!llvm.ptr, i32, !mpi.datatype)
    %i32 = "mpi.get_dtype"() {dtype = i32} : () -> !mpi.datatype
}) : () -> ()


// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   %ref = "memref.alloc"() {alignment = 32 : i64, operandSegmentSizes = array<i32: 0, 0>} : () -> memref<100x14x14xf64>
// CHECK-NEXT:   %buff = "memref.extract_aligned_pointer_as_index"(%ref) : (memref<100x14x14xf64>) -> index
// CHECK-NEXT:   %buff1 = "arith.index_cast"(%buff) : (index) -> i64
// CHECK-NEXT:   %buff2 = "llvm.inttoptr"(%buff1) : (i64) -> !llvm.ptr
// CHECK-NEXT:   %buff3 = "arith.constant"() {value = 128 : i32} : () -> i32
// CHECK-NEXT:   %buff4 = "arith.constant"() {value = 1275070475 : i32} : () -> i32
// CHECK-NEXT:   %0 = "arith.constant"() {value = 1275069445 : i32} : () -> i32
// CHECK-NEXT: }) : () -> ()
