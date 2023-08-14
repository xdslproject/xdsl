// RUN: xdsl-opt %s | xdsl-opt
"builtin.module"() ({
    func.func @mpi_example(%ref : memref<100xf32>, %dest : i32, %tag : i32) {
        mpi.init

        %rank = mpi.comm.rank : i32

        %ptr, %size, %typ = "mpi.unwrap_memref"(%ref) : (memref<100xf32>) -> (!llvm.ptr, i32, !mpi.datatype)

        %typ_i32 = mpi.get_dtype of i32 -> !mpi.datatype

        "mpi.send"(%ptr, %size, %typ, %dest, %tag) : (!llvm.ptr, i32, !mpi.datatype, i32, i32) -> ()

        %status = "mpi.recv"(%ptr, %size, %typ, %dest, %tag) : (!llvm.ptr, i32, !mpi.datatype, i32, i32) -> !mpi.status

        %f = mpi.status.get %status[#mpi.status.field<MPI_SOURCE>] : !mpi.status -> i32

        %f = mpi.status.get %status[#mpi.status.field<MPI_TAG>] : !mpi.status -> i32

        %f = mpi.status.get %status[#mpi.status.field<MPI_ERROR>] : !mpi.status -> i32

        mpi.finalize

        func.return
    }
}) : () -> ()

