// RUN: xdsl-opt %s | xdsl-opt | filecheck %s
// RUN: xdsl-opt %s | xdsl-opt -p "lower-mpi{vendor=mpich}"| filecheck %s --check-prefix=MPICH
// RUN: xdsl-opt %s | xdsl-opt -p "lower-mpi{vendor=ompi}" | filecheck %s --check-prefix=OMPI

"builtin.module"() ({
    func.func @mpi_example(%ref : memref<100xf32>, %dest : i32, %tag : i32) {
        mpi.init

        %rank = mpi.comm.rank : i32

        %ptr, %size, %typ = "mpi.mlir.unwrap_memref"(%ref) : (memref<100xf32>) -> (!llvm.ptr, i32, !mpi.datatype)

        %typ_i32 = mpi.mlir.get_dtype of i32 -> !mpi.datatype

        "mpi.send"(%ptr, %size, %typ, %dest, %tag) : (!llvm.ptr, i32, !mpi.datatype, i32, i32) -> ()

        "mpi.recv"(%ptr, %size, %typ, %dest, %tag) : (!llvm.ptr, i32, !mpi.datatype, i32, i32) -> ()

        mpi.finalize

        func.return
    }
}) : () -> ()


// CHECK:      mpi.init
// CHECK-NEXT: %rank = mpi.comm.rank : i32
// CHECK-NEXT: %ptr, %size, %typ = "mpi.mlir.unwrap_memref"(%ref) : (memref<100xf32>) -> (!llvm.ptr, i32, !mpi.datatype)
// CHECK-NEXT: %typ_i32 = mpi.mlir.get_dtype of i32 -> !mpi.datatype
// CHECK-NEXT: "mpi.send"(%ptr, %size, %typ, %dest, %tag) : (!llvm.ptr, i32, !mpi.datatype, i32, i32) -> ()
// CHECK-NEXT: "mpi.recv"(%ptr, %size, %typ, %dest, %tag) : (!llvm.ptr, i32, !mpi.datatype, i32, i32) -> ()
// CHECK-NEXT: mpi.finalize
// CHECK-NEXT: func.return


// OMPI: builtin.module {
// OMPI-NEXT:   func.func @mpi_example(%ref : memref<100xf32>, %dest : i32, %tag : i32) {
// OMPI-NEXT:     %0 = "llvm.mlir.null"() : () -> !llvm.ptr
// OMPI-NEXT:     %1 = "llvm.call"(%0, %0) {"callee" = @MPI_Init, "fastmathFlags" = #llvm.fastmath<none>} : (!llvm.ptr, !llvm.ptr) -> i32
// OMPI-NEXT:     %rank = "llvm.mlir.addressof"() {"global_name" = @ompi_mpi_comm_world} : () -> !llvm.ptr
// OMPI-NEXT:     %rank_1 = arith.constant 1 : i64
// OMPI-NEXT:     %rank_2 = "llvm.alloca"(%rank_1) {"alignment" = 32 : i64, "elem_type" = i32} : (i64) -> !llvm.ptr
// OMPI-NEXT:     %rank_3 = "llvm.call"(%rank, %rank_2) {"callee" = @MPI_Comm_rank, "fastmathFlags" = #llvm.fastmath<none>} : (!llvm.ptr, !llvm.ptr) -> i32
// OMPI-NEXT:     %rank_4 = "llvm.load"(%rank_2) : (!llvm.ptr) -> i32
// OMPI-NEXT:     %ptr = "llvm.mlir.null"() : () -> !llvm.ptr
// OMPI-NEXT:     %ptr_1 = arith.constant 100 : i32
// OMPI-NEXT:     %ptr_2 = "llvm.mlir.addressof"() {"global_name" = @ompi_mpi_float} : () -> !llvm.ptr
// OMPI-NEXT:     %typ_i32 = "llvm.mlir.addressof"() {"global_name" = @ompi_mpi_int} : () -> !llvm.ptr
// OMPI-NEXT:     %2 = "llvm.mlir.addressof"() {"global_name" = @ompi_mpi_comm_world} : () -> !llvm.ptr
// OMPI-NEXT:     %3 = "llvm.call"(%ptr, %ptr_1, %ptr_2, %dest, %tag, %2) {"callee" = @MPI_Send, "fastmathFlags" = #llvm.fastmath<none>} : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> i32
// OMPI-NEXT:     %4 = "llvm.mlir.addressof"() {"global_name" = @ompi_mpi_comm_world} : () -> !llvm.ptr
// OMPI-NEXT:     %5 = "llvm.mlir.null"() : () -> !llvm.ptr
// OMPI-NEXT:     %6 = "llvm.call"(%ptr, %ptr_1, %ptr_2, %dest, %tag, %4, %5) {"callee" = @MPI_Send, "fastmathFlags" = #llvm.fastmath<none>} : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// OMPI-NEXT:     %7 = "llvm.call"() {"callee" = @MPI_Finalize, "fastmathFlags" = #llvm.fastmath<none>} : () -> i32
// OMPI-NEXT:     func.return
// OMPI-NEXT:   }
// OMPI-NEXT:   "llvm.mlir.global"() ({
// OMPI-NEXT:   }) {"global_type" = i32, "sym_name" = "ompi_mpi_comm_world", "linkage" = #llvm.linkage<"external">, "addr_space" = 0 : i32} : () -> ()
// OMPI-NEXT:   "llvm.mlir.global"() ({
// OMPI-NEXT:   }) {"global_type" = i32, "sym_name" = "ompi_mpi_float", "linkage" = #llvm.linkage<"external">, "addr_space" = 0 : i32} : () -> ()
// OMPI-NEXT:   "llvm.mlir.global"() ({
// OMPI-NEXT:   }) {"global_type" = i32, "sym_name" = "ompi_mpi_int", "linkage" = #llvm.linkage<"external">, "addr_space" = 0 : i32} : () -> ()
// OMPI-NEXT:   "llvm.func"() ({
// OMPI-NEXT:   }) {"sym_name" = "MPI_Init", "function_type" = !llvm.func<i32 (!llvm.ptr, !llvm.ptr)>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// OMPI-NEXT:   "llvm.func"() ({
// OMPI-NEXT:   }) {"sym_name" = "MPI_Comm_rank", "function_type" = !llvm.func<i32 (!llvm.ptr, !llvm.ptr)>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// OMPI-NEXT:   "llvm.func"() ({
// OMPI-NEXT:   }) {"sym_name" = "MPI_Send", "function_type" = !llvm.func<i32 (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr)>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// OMPI-NEXT:   "llvm.func"() ({
// OMPI-NEXT:   }) {"sym_name" = "MPI_Finalize", "function_type" = !llvm.func<i32 ()>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// OMPI-NEXT: }


// MPICH: builtin.module {
// MPICH-NEXT:   func.func @mpi_example(%ref : memref<100xf32>, %dest : i32, %tag : i32) {
// MPICH-NEXT:     %0 = "llvm.mlir.null"() : () -> !llvm.ptr
// MPICH-NEXT:     %1 = "llvm.call"(%0, %0) {"callee" = @MPI_Init, "fastmathFlags" = #llvm.fastmath<none>} : (!llvm.ptr, !llvm.ptr) -> i32
// MPICH-NEXT:     %rank = arith.constant 1140850688 : i32
// MPICH-NEXT:     %rank_1 = arith.constant 1 : i64
// MPICH-NEXT:     %rank_2 = "llvm.alloca"(%rank_1) {"alignment" = 32 : i64, "elem_type" = i32} : (i64) -> !llvm.ptr
// MPICH-NEXT:     %rank_3 = "llvm.call"(%rank, %rank_2) {"callee" = @MPI_Comm_rank, "fastmathFlags" = #llvm.fastmath<none>} : (i32, !llvm.ptr) -> i32
// MPICH-NEXT:     %rank_4 = "llvm.load"(%rank_2) : (!llvm.ptr) -> i32
// MPICH-NEXT:     %ptr = "llvm.mlir.null"() : () -> !llvm.ptr
// MPICH-NEXT:     %ptr_1 = arith.constant 100 : i32
// MPICH-NEXT:     %ptr_2 = arith.constant 1275069450 : i32
// MPICH-NEXT:     %typ_i32 = arith.constant 1275069445 : i32
// MPICH-NEXT:     %2 = arith.constant 1140850688 : i32
// MPICH-NEXT:     %3 = "llvm.call"(%ptr, %ptr_1, %ptr_2, %dest, %tag, %2) {"callee" = @MPI_Send, "fastmathFlags" = #llvm.fastmath<none>} : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
// MPICH-NEXT:     %4 = arith.constant 1140850688 : i32
// MPICH-NEXT:     %5 = arith.constant 1 : i32
// MPICH-NEXT:     %6 = "llvm.call"(%ptr, %ptr_1, %ptr_2, %dest, %tag, %4, %5) {"callee" = @MPI_Send, "fastmathFlags" = #llvm.fastmath<none>} : (!llvm.ptr, i32, i32, i32, i32, i32, i32) -> i32
// MPICH-NEXT:     %7 = "llvm.call"() {"callee" = @MPI_Finalize, "fastmathFlags" = #llvm.fastmath<none>} : () -> i32
// MPICH-NEXT:     func.return
// MPICH-NEXT:   }
// MPICH-NEXT:   "llvm.func"() ({
// MPICH-NEXT:   }) {"sym_name" = "MPI_Init", "function_type" = !llvm.func<i32 (!llvm.ptr, !llvm.ptr)>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// MPICH-NEXT:   "llvm.func"() ({
// MPICH-NEXT:   }) {"sym_name" = "MPI_Comm_rank", "function_type" = !llvm.func<i32 (i32, !llvm.ptr)>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// MPICH-NEXT:   "llvm.func"() ({
// MPICH-NEXT:   }) {"sym_name" = "MPI_Send", "function_type" = !llvm.func<i32 (!llvm.ptr, i32, i32, i32, i32, i32)>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// MPICH-NEXT:   "llvm.func"() ({
// MPICH-NEXT:   }) {"sym_name" = "MPI_Finalize", "function_type" = !llvm.func<i32 ()>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// MPICH-NEXT: }
