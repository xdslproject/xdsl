// RUN: xdsl-opt %s -p lower-mpi | mlir-opt --convert-func-to-llvm --expand-strided-metadata --normalize-memrefs --memref-expand --fold-memref-alias-ops --finalize-memref-to-llvm --reconcile-unrealized-casts | filecheck %s

"builtin.module"() ({
  "func.func"() ({
    "mpi.init"() : () -> ()
    %rank = "mpi.comm.rank"() : () -> i32
    %cst0 = "arith.constant"() {"value" = 0 : i32} : () -> i32
    %cst1 = "arith.constant"() {"value" = 2 : i32} : () -> i32
    %cst2 = "arith.constant"() {"value" = 2 : i32} : () -> i32
    %reqs = "mpi.allocate"(%cst2) {dtype = !mpi.request} : (i32) -> !mpi.vector<!mpi.request>
    %rank_is_zero = "arith.cmpi"(%rank, %cst0) {"predicate" = 0 : i64} : (i32, i32) -> i1
    %ref = "memref.alloc"() {"alignment" = 32 : i64, operandSegmentSizes = array<i32: 0, 0>} : () -> memref<100x14x14xf64>
    %tag = "arith.constant"() {"value" = 1 : i32} : () -> i32
    %buff, %count, %dtype = "mpi.unwrap_memref"(%ref) : (memref<100x14x14xf64>) -> (!llvm.ptr, i32, !mpi.datatype)
    "scf.if"(%rank_is_zero) ({
      %dest = "arith.constant"() {"value" = 1 : i32} : () -> i32
      %req = "mpi.vector_get"(%reqs, %cst0) : (!mpi.vector<!mpi.request>, i32) -> !mpi.request
      "mpi.isend"(%buff, %count, %dtype, %dest, %tag, %req) : (!llvm.ptr, i32, !mpi.datatype, i32, i32, !mpi.request) -> ()
      "mpi.wait"(%req) : (!mpi.request) -> ()
      "scf.yield"() : () -> ()
    }, {
      %source = "arith.constant"() {"value" = 0 : i32} : () -> i32
      %req = "mpi.vector_get"(%reqs, %cst0) : (!mpi.vector<!mpi.request>, i32) -> !mpi.request
      "mpi.irecv"(%buff, %count, %dtype, %source, %tag, %req) {"tag" = 1 : i32} : (!llvm.ptr, i32, !mpi.datatype, i32, i32, !mpi.request) -> ()
      %status = "mpi.wait"(%req) : (!mpi.request) -> !mpi.status
      "scf.yield"() : () -> ()
    }) : (i1) -> ()
    "mpi.finalize"() : () -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "main", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// we don't really care about the whole structure, we just want to make sure mlir-opt can lower all this down to llvm

// CHECK: llvm.call @MPI_Init({{%\S+}}, {{%\S+}}) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.call @MPI_Comm_rank({{%\S+}}, {{%\S+}}) : (i32, !llvm.ptr) -> i32
// CHECK: llvm.call @MPI_Isend({{%\S+}}, {{%\S+}}, {{%\S+}}, {{%\S+}}, {{%\S+}}, {{%\S+}}, {{%\S+}}) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK: llvm.call @MPI_Wait({{%\S+}}, {{%\S+}}) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.call @MPI_Irecv({{%\S+}}, {{%\S+}}, {{%\S+}}, {{%\S+}}, {{%\S+}}, {{%\S+}}, {{%\S+}}) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK: llvm.call @MPI_Wait({{%\S+}}, {{%\S+}}) : (!llvm.ptr, !llvm.ptr) -> i32

// also check that external funcs were declared correctly:

// CHECK: llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
// CHECK: llvm.func @MPI_Comm_rank(i32, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
// CHECK: llvm.func @MPI_Isend(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
// CHECK: llvm.func @MPI_Wait(!llvm.ptr, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
// CHECK: llvm.func @MPI_Irecv(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
// CHECK: llvm.func @MPI_Finalize() -> i32 attributes {sym_visibility = "private"}
