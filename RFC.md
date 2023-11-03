# MPI Dialect: RFC

This dialect models the Message Passing Interface (MPI), version 4.0. It is meant to serve as an interface dialect that is targeted by higher-level dialects. The MPI dialect itself can be lowered to multiple MPI implementations and hide differences in ABI. The dialect models the functions of the MPI specification as close to 1:1 as possible while preserving SSA value semantics where it makes sense, and uses `memref` types instead of bare pointers.

For an in-depth documentation of the MPI library interface, please refer to official documentation such as the [OpenMPI online documentation](https://www.open-mpi.org/doc/current/). Relevant parts of the documentation are linked throughout this RFC.

This RFC does not cover all of the MPI specification, it will instead focus on the following feature sets:

| Feature | State | Comment |
|---------|-------|-----------|
| Blocking send/recv | PR Ready | Presented at ODM |
| Nonblocking send/recv | Example IR | Validated internally |
| Communicators | Example IR | |
| Collectives | Example IR | |
| Lowering | Example IR, POC | |
| MPI Error codes | Example IR | |
| Handling MPI Status | Example IR| |

According to [A large-scale study of MPI usage in open-source HPC applications](https://dl.acm.org/doi/10.1145/3295500.3356176), a small subset of all MPI calls make up the majority of MPI uses. The subset presented in this RFC provides good coverage of large parts of real-world HPC MPI usecases. This does not mean however, that features absent from this RFC are excluded from the MPI dialect. Additionally, features outlined in this RFC are not necessarily planned to be added to the dialect in the near future. It is instead intended to explore and show the decisions made while modelling MPI as an MLIR dialect and to verify that they make sense and are able to represent real HPC programs.

A collection of open questions is posed at the bottom of this RFC.

# Blocking  Communication

These are the simplest building blocks of MPI, our initial PR contains a simple
synchronous send/receive, init, finalise, and an operation to obtain
the processes rank:

```mlir
func.func @mpi_test(%ref : memref<100xf32>) -> () {
    mpi.init

    %rank = mpi.comm_rank : i32

    mpi.send(%ref, %rank, %tag) : memref<100xf32>, i32, i32

    mpi.recv(%ref, %rank, %tag) : memref<100xf32>, i32, i32

    mpi.finalize

    func.return
}
```

For a more detailed look at this initial set of operations see 
[the PR](https://github.com/llvm/llvm-project/pull/68892) which provides the output of `mlir-tblgen -gen-dialect-doc`.

The decision to model MPIs pointer+size+type as MLIR `memref`s was made because we felt that the dialect would fit better into the existing ecosystem of MLIR dialects.
# Non-blocking Communication

For non-blocking communication, a new datatype `!mpi.request`  is introduced. This is directly equivalent to the `MPI_Request` type defined by MPI.

Since `MPI_Request`s are mutable objects that are always passed by reference, we decide to model them in memrefs and pass them as memref+index. This is consistent with how they are most often used in actual HPC programs (i.e. a stack-allocated array of `MPI_Request` objects).

With this, the nonblocking version of the blocking example above looks like this:

```mlir
func.func @mpi_test(%ref : memref<100xf32>) -> () {
    mpi.init

    %rank = mpi.comm_rank : i32

    %requests = memref.alloca() : memref<2x!mpi.request>

    mpi.isend (%ref, %rank, %rank) as %requests[0] : memref<100xf32>, i32, i32, memref<2x!mpi.request>

    mpi.irecv (%ref, %rank, %rank) as %requests[1] : memref<100xf32>, i32, i32, memref<2x!mpi.request>

    // either waiting on a single one:
    %status = mpi.wait %requests[0] : memref<2x!mpi.request> -> !mpi.status

    // issue a waitall for all requests
    mpi.waitall %requests : memref<2x!mpi.request>

    mpi.finalize

    func.return
}
```

Implementing [MPI_Wait](https://www.open-mpi.org/doc/v4.1/man3/MPI_Wait.3.php), [MPI_Waitany](https://www.open-mpi.org/doc/v4.1/man3/MPI_Waitany.3.php), [MPI_Test](https://www.open-mpi.org/doc/v4.1/man3/MPI_Test.3.php), or [MPI_Testany](https://www.open-mpi.org/doc/v4.1/man3/MPI_Testany.3.php) would be straightforward when modelled this way.

### `MPI_REQUEST_NULL`:

Modelling `MPI_REQUEST_NULL` would be done similar to the way `nullptr`s are handled in the llvm dialect. Since this is an immutable constant value, we are okay with it existing outside of a memref.

```mlir
%requests = memref.alloca() : memref<2x!mpi.request>
%null_req = mpi.request_null : -> !mpi.request
memref.store %null_req %request[%c0] : memref<2x!mpi.request>
```

# Communicators

MPI communicators are at the heart of many HPC programs. They give rise to interesting structures and allow to abstract away complexity in selecting communication partners as well as providing guaranteed separation for library code. We introduce the `!mpi.comm` type to model communicators. As an example, here is how we imagine `MPI_Comm_split` and `MPI_Comm_dup` to work:

```mlir
%comm_world = mpi.comm_world : !mpi.comm

%split = mpi.comm_split %comm_world by %color, %key : (!mpi.comm, i32, i32) -> !mpi.comm

%dup = mpi.comm_dup %split : !mpi.comm -> !mpi.comm

// other communicator constants can be modelled like this:
%comm_null = mpi.comm_null : !mpi.comm
%comm_self = mpi.comm_self : !mpi.comm
```

The patch that introduces communicators would add an `!mpi.comm` argument to every operation that requires a communicator.
### Case Study: Cartesian Topology

We also want to look at how we would model Cartesian communicators:

```mlir
%comm_world = mpi.comm_world : !mpi.comm
%nodes = mpi.comm_size %comm_world : !mpi.comm -> i32

%dims = memref.alloca : memref<3xi32>
// initialize to [0,0,2]
memref.store %c0, %dims[0] : memref<3xi32>
memref.store %c0, %dims[1] : memref<3xi32>
memref.store %c2, %dims[2] : memref<3xi32>

// int MPI_Dims_create(int nnodes, int ndims, int dims[])
// ndims will be inferred from the memref size.
// results will be written back into %dims
mpi.dims_create %nodes, %dims : %i32, memref<3x132>

// periods = [true, true, false]
%periods = memref.alloca : memref<3xi32>
// memref initialization left out for brevity

%reorder = arith.constant true : i1

%comm_cart = mpi.cart_create %comm_world, %dims, %periods, %reorder : (!mpi.comm, memref<3xi32>, memref<3xi32>, i1) -> !mpi.comm
```

Here are the documentation pages of OpenMPI for reference: [MPI_Comm_size](https://www-lb.open-mpi.org/doc/v4.1/man3/MPI_Comm_size.3.php), [MPI_Dims_create](https://www-lb.open-mpi.org/doc/v4.1/man3/MPI_Dims_create.3.php) and [MPI_Cart_create](https://www-lb.open-mpi.org/doc/v4.1/man3/MPI_Cart_create.3.php). Using the created Cartesian communicator would look like this:

```mlir
// get number of dims
%dims = mpi.cartdim_get %comm_cart : !mpi.comm -> i32

// allocate a memref to hold cartesian coordinates:
%coords = memref.alloca(%dims) : memref<?xi32>

// get rank in communicator
%rank = mpi.comm_rank %comm_cart : !mpi.comm -> i32

// translate rank to cartesian coordinates:
mpi.cart_coords %comm_cart, %coords : !mpi.comm, memref<?xi32>

// update rank
mock.calc_dest_coords %coords : memref<?xi32>

// translate back into dest rank:
%rank = mpi.cart_rank %comm_cart, %coords : !mpi.comm, memref<?xi32> -> i32
```

This uses [MPI_Cartdim_get](https://www-lb.open-mpi.org/doc/v4.1/man3/MPI_Cartdim_get.3.php), [MPI_Comm_rank](https://www-lb.open-mpi.org/doc/v4.1/man3/MPI_Comm_rank.3.php), [MPI_Cart_coords](https://www-lb.open-mpi.org/doc/v4.1/man3/MPI_Cart_coords.3.php) and [MPI_Cart_rank](https://www-lb.open-mpi.org/doc/v4.1/man3/MPI_Cart_rank.3.php).

*Notes:*
 - `MPI_Cart_rank` expects the array to have exactly `ndims` elements, which we can't universally verify at compile time.

We hope that this illustrates that the concept of MPI Communicators can be broadly mapped to MLIR in a consistent fashion.

One can see that mapping `MPI_Group` operations can be done in an analogous fashion to topologies.

# Collectives / Operations

The easiest case of an [MPI_Allreduce](https://www-lb.open-mpi.org/doc/v4.1/man3/MPI_Allreduce.3.php) using `MPI_SUM` can be modelled like this:

```mlir
%sum = mpi.op sum : !mpi.op
%outref = memref.alloc() : !memref<100xf32>

mpi.allreduce %ref with %sum into %outref on %my_comm : memref<100xf32>

// with MPI_IN_PLACE, replace `into` $dest with `in_place`
mpi.allreduce %ref with %sum in_place on %my_comm : memref<100xf32>
```

A simple [MPI_Reduce](https://www-lb.open-mpi.org/doc/v4.1/man3/MPI_Reduce.3.php) poses an additional challenge, as the result buffer is only written to on rank 0, meaning we would
not want to allocate a full memref on each rank. Our idea is to allow unsized memref arguments on the destination.

```mlir
%rank = mpi.comm_rank %my_comm : i32
%root = arith.constant 0 : i32
%is_root = arith.cmpi eq, %rank, %root : i32

// allocate memref only on root rank
%dest = scf.if %is_root -> (memref<?xf32>) {
    %ref = memref.alloc() : memref<100xf32>
    %unsized = memref.cast %ref : memref<100xf32> to memref<?xf32>
    scf.yield %unsized : memref<?xf32>
} else {
    %ref_empty = memref.alloc() : memref<0xf32>
    %unsized_empty = memref.cast %ref_empty : memref<0xf32> to memref<?xf32>
    scf.yield %unsized_empty : memref<?xf32>
}

mpi.reduce %data with %sum into %dest rank %rank on %my_comm : memref<100xf32>, !mpi.op, memref<?xf32>, i32, !mpi.comm

// in-place
mpi.reduce %data with %sum in_place rank %rank on %my_comm : memref<100xf32>, !mpi.op, i32, !mpi.comm

scf.if %is_root {
    %sized = memref.cast %dest : memref<?xf32> to memref<100xf32>
    // use data
}
```

The conditional allocation could be provided in a helper operation:

```mlir
%dest_ref = mpi.allocate_on_rank %my_rank, %rank, memref<100xf32> -> memref<?xf32>
```

Defining custom `MPI_Op`s using [MPI_Op_create](https://www-lb.open-mpi.org/doc/v4.1/man3/MPI_Op_create.3.php):

```mlir
// generates an operator with validity for a single datatype:
func.func @mpi_custom_op (%in: memref<?xf32>, %inout: memref<?xf32>) {
    // runtime assert could be inserted into this function
    // compute operator
}

%commute = arith.constant 1: i32

%custom_op = mpi.op_create @mpi_custom_op, %commute : i32 -> !mpi.op
```

MPI requires the following format for user supplied functions:

```C
typedef void MPI_User_function(
    void *invec, 
    void *inoutvec,
    int *len,
    MPI_Datatype *datatype
);
```

Modelling and inspecting `MPI_Datatype` at runtime as part of a custom op is currently not part of this RFC, but could be added if it is actually needed.

# Handling `MPI_Status`

In order to handle MPI Status, we would introduce an optional result value of type `!mpi.status`. The `MPI_Status` is defined to be a struct with at least three fields (`MPI_SOURCE`, `MPI_TAG` and `MPI_ERROR`). Additionally, one can get the number of elements sent the from a status object using the `MPI_Get_count` function. We provide an accessor operation for these fields and additional operations for `MPI_Get_count`.

```mlir
%status = mpi.send (%ref, %rank, %tag) : (memref<100xf32>, i32, i32) -> !mpi.status

// access struct members:
%source = mpi.status_get_field %status[MPI_SOURCE] : !mpi.status -> i32
%tag = mpi.status_get_field %status[MPI_TAG] : !mpi.status -> i32
%err = mpi.status_get_field %status[MPI_ERROR] : !mpi.status -> !mpi.retval

// using the MPI_Get_count function to access get the element count:
%count = mpi.get_count %status : !mpi.status -> i32
```

# Lowering and Differences in ABI

This part gets into the ABI differences between implementation. We highly recommend the paper on [MPI Application Binary Interface Standardization](https://arxiv.org/pdf/2308.11214.pdf) as a primer for this section.

We have implemented an example showing off how we lower our initial patch to both MPICH and OpenMPI style ABIs (using xDSL for quick prototyping). We target the llvm dialect directly because we need access to low-level concepts like pointers, structs, etc. We hope that the messy output below is enough argument in favour of introducing the MPI dialect abstraction:

```mlir
// RUN: xdsl-opt %s | xdsl-opt -p "lower-mpi{vendor=mpich}"| filecheck %s --check-prefix=MPICH
// RUN: xdsl-opt %s | xdsl-opt -p "lower-mpi{vendor=ompi}" | filecheck %s --check-prefix=OMPI

"builtin.module"() ({
    func.func @mpi_example(%ref : memref<100xf32>, %dest : i32, %tag : i32) {
        mpi.init

        %rank = mpi.comm.rank : i32

        "mpi.send"(%ref, %dest, %tag) : (memref<100xf32>, i32, i32) -> ()

        "mpi.recv"(%ref, %dest, %tag) : (memref<100xf32>, i32, i32) -> ()

        mpi.finalize

        func.return
    }
}) : () -> ()


// Lowering to OpenMPI's opaque struct pointers:

// OMPI:      builtin.module {
// OMPI-NEXT:   func.func @mpi_example(%ref : memref<100xf32>, %dest : i32, %tag : i32) {
// OMPI-NEXT:     %0 = "llvm.mlir.null"() : () -> !llvm.ptr
// OMPI-NEXT:     %1 = "llvm.call"(%0, %0) {"callee" = @MPI_Init, "fastmathFlags" = #llvm.fastmath<none>} : (!llvm.ptr, !llvm.ptr) -> i32
// OMPI-NEXT:     %rank = "llvm.mlir.addressof"() {"global_name" = @ompi_mpi_comm_world} : () -> !llvm.ptr
// OMPI-NEXT:     %rank_1 = arith.constant 1 : i64
// OMPI-NEXT:     %rank_2 = "llvm.alloca"(%rank_1) {"alignment" = 32 : i64, "elem_type" = i32} : (i64) -> !llvm.ptr
// OMPI-NEXT:     %rank_3 = "llvm.call"(%rank, %rank_2) {"callee" = @MPI_Comm_rank, "fastmathFlags" = #llvm.fastmath<none>} : (!llvm.ptr, !llvm.ptr) -> i32
// OMPI-NEXT:     %rank_4 = "llvm.load"(%rank_2) : (!llvm.ptr) -> i32
// OMPI-NEXT:     %2 = "memref.extract_aligned_pointer_as_index"(%ref) : (memref<100xf32>) -> index
// OMPI-NEXT:     %3 = "arith.index_cast"(%2) : (index) -> i64
// OMPI-NEXT:     %4 = "llvm.inttoptr"(%3) : (i64) -> !llvm.ptr
// OMPI-NEXT:     %5 = arith.constant 100 : i32
// OMPI-NEXT:     %6 = "llvm.mlir.addressof"() {"global_name" = @ompi_mpi_float} : () -> !llvm.ptr
// OMPI-NEXT:     %7 = "llvm.mlir.addressof"() {"global_name" = @ompi_mpi_comm_world} : () -> !llvm.ptr
// OMPI-NEXT:     %8 = "llvm.call"(%4, %5, %6, %dest, %tag, %7) {"callee" = @MPI_Send, "fastmathFlags" = #llvm.fastmath<none>} : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr) -> i32
// OMPI-NEXT:     %9 = "memref.extract_aligned_pointer_as_index"(%ref) : (memref<100xf32>) -> index
// OMPI-NEXT:     %10 = "arith.index_cast"(%9) : (index) -> i64
// OMPI-NEXT:     %11 = "llvm.inttoptr"(%10) : (i64) -> !llvm.ptr
// OMPI-NEXT:     %12 = arith.constant 100 : i32
// OMPI-NEXT:     %13 = "llvm.mlir.addressof"() {"global_name" = @ompi_mpi_float} : () -> !llvm.ptr
// OMPI-NEXT:     %14 = "llvm.mlir.addressof"() {"global_name" = @ompi_mpi_comm_world} : () -> !llvm.ptr
// OMPI-NEXT:     %15 = "llvm.mlir.null"() : () -> !llvm.ptr
// OMPI-NEXT:     %16 = "llvm.call"(%11, %12, %13, %dest, %tag, %14, %15) {"callee" = @MPI_Recv, "fastmathFlags" = #llvm.fastmath<none>} : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// OMPI-NEXT:     %17 = "llvm.call"() {"callee" = @MPI_Finalize, "fastmathFlags" = #llvm.fastmath<none>} : () -> i32
// OMPI-NEXT:     func.return
// OMPI-NEXT:   }
// OMPI-NEXT:   "llvm.mlir.global"() ({
// OMPI-NEXT:   }) {"global_type" = i32, "sym_name" = "ompi_mpi_comm_world", "linkage" = #llvm.linkage<"external">, "addr_space" = 0 : i32} : () -> ()
// OMPI-NEXT:   "llvm.mlir.global"() ({
// OMPI-NEXT:   }) {"global_type" = i32, "sym_name" = "ompi_mpi_float", "linkage" = #llvm.linkage<"external">, "addr_space" = 0 : i32} : () -> ()
// OMPI-NEXT:   "llvm.func"() ({
// OMPI-NEXT:   }) {"sym_name" = "MPI_Init", "function_type" = !llvm.func<i32 (!llvm.ptr, !llvm.ptr)>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// OMPI-NEXT:   "llvm.func"() ({
// OMPI-NEXT:   }) {"sym_name" = "MPI_Comm_rank", "function_type" = !llvm.func<i32 (!llvm.ptr, !llvm.ptr)>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// OMPI-NEXT:   "llvm.func"() ({
// OMPI-NEXT:   }) {"sym_name" = "MPI_Send", "function_type" = !llvm.func<i32 (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr)>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// OMPI-NEXT:   "llvm.func"() ({
// OMPI-NEXT:   }) {"sym_name" = "MPI_Recv", "function_type" = !llvm.func<i32 (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr)>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// OMPI-NEXT:   "llvm.func"() ({
// OMPI-NEXT:   }) {"sym_name" = "MPI_Finalize", "function_type" = !llvm.func<i32 ()>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// OMPI-NEXT: }


// Lowering to MPICHs integer constants:

// MPICH:      builtin.module {
// MPICH-NEXT:   func.func @mpi_example(%ref : memref<100xf32>, %dest : i32, %tag : i32) {
// MPICH-NEXT:     %0 = "llvm.mlir.null"() : () -> !llvm.ptr
// MPICH-NEXT:     %1 = "llvm.call"(%0, %0) {"callee" = @MPI_Init, "fastmathFlags" = #llvm.fastmath<none>} : (!llvm.ptr, !llvm.ptr) -> i32
// MPICH-NEXT:     %rank = arith.constant 1140850688 : i32
// MPICH-NEXT:     %rank_1 = arith.constant 1 : i64
// MPICH-NEXT:     %rank_2 = "llvm.alloca"(%rank_1) {"alignment" = 32 : i64, "elem_type" = i32} : (i64) -> !llvm.ptr
// MPICH-NEXT:     %rank_3 = "llvm.call"(%rank, %rank_2) {"callee" = @MPI_Comm_rank, "fastmathFlags" = #llvm.fastmath<none>} : (i32, !llvm.ptr) -> i32
// MPICH-NEXT:     %rank_4 = "llvm.load"(%rank_2) : (!llvm.ptr) -> i32
// MPICH-NEXT:     %2 = "memref.extract_aligned_pointer_as_index"(%ref) : (memref<100xf32>) -> index
// MPICH-NEXT:     %3 = "arith.index_cast"(%2) : (index) -> i64
// MPICH-NEXT:     %4 = "llvm.inttoptr"(%3) : (i64) -> !llvm.ptr
// MPICH-NEXT:     %5 = arith.constant 100 : i32
// MPICH-NEXT:     %6 = arith.constant 1275069450 : i32
// MPICH-NEXT:     %7 = arith.constant 1140850688 : i32
// MPICH-NEXT:     %8 = "llvm.call"(%4, %5, %6, %dest, %tag, %7) {"callee" = @MPI_Send, "fastmathFlags" = #llvm.fastmath<none>} : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
// MPICH-NEXT:     %9 = "memref.extract_aligned_pointer_as_index"(%ref) : (memref<100xf32>) -> index
// MPICH-NEXT:     %10 = "arith.index_cast"(%9) : (index) -> i64
// MPICH-NEXT:     %11 = "llvm.inttoptr"(%10) : (i64) -> !llvm.ptr
// MPICH-NEXT:     %12 = arith.constant 100 : i32
// MPICH-NEXT:     %13 = arith.constant 1275069450 : i32
// MPICH-NEXT:     %14 = arith.constant 1140850688 : i32
// MPICH-NEXT:     %15 = arith.constant 1 : i32
// MPICH-NEXT:     %16 = "llvm.call"(%11, %12, %13, %dest, %tag, %14, %15) {"callee" = @MPI_Recv, "fastmathFlags" = #llvm.fastmath<none>} : (!llvm.ptr, i32, i32, i32, i32, i32, i32) -> i32
// MPICH-NEXT:     %17 = "llvm.call"() {"callee" = @MPI_Finalize, "fastmathFlags" = #llvm.fastmath<none>} : () -> i32
// MPICH-NEXT:     func.return
// MPICH-NEXT:   }
// MPICH-NEXT:   "llvm.func"() ({
// MPICH-NEXT:   }) {"sym_name" = "MPI_Init", "function_type" = !llvm.func<i32 (!llvm.ptr, !llvm.ptr)>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// MPICH-NEXT:   "llvm.func"() ({
// MPICH-NEXT:   }) {"sym_name" = "MPI_Comm_rank", "function_type" = !llvm.func<i32 (i32, !llvm.ptr)>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// MPICH-NEXT:   "llvm.func"() ({
// MPICH-NEXT:   }) {"sym_name" = "MPI_Send", "function_type" = !llvm.func<i32 (!llvm.ptr, i32, i32, i32, i32, i32)>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// MPICH-NEXT:   "llvm.func"() ({
// MPICH-NEXT:   }) {"sym_name" = "MPI_Recv", "function_type" = !llvm.func<i32 (!llvm.ptr, i32, i32, i32, i32, i32, i32)>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// MPICH-NEXT:   "llvm.func"() ({
// MPICH-NEXT:   }) {"sym_name" = "MPI_Finalize", "function_type" = !llvm.func<i32 ()>, "CConv" = #llvm.cconv<ccc>, "linkage" = #llvm.linkage<"external">, "visibility_" = 0 : i64} : () -> ()
// MPICH-NEXT: }

```

We slightly prefer supporting to multiple implementations through a toggle in the lowering instead of an MLIR runtime but don't want to rule out anything yet. The ABI standardisation efforts put forth by Hammond et al. hint at a more unified landscape in the future.
# MPI Error Codes

Almost all MPI functions return error codes (C `int`) (which are often ignored). We propose to add an optional result to all operations that can return error codes. This result value will be of type `!mpi.retval`, that can be queried against various error codes:

```mlir
%err = mpi.send ...

// Check if returned value is MPI_SUCCESS
%is_success = mpi.retval_check %err = MPI_SUCCESS : !mpi.retval -> i1
%is_err_in_stat = mpi.retval_check %err = MPI_ERR_IN_STATUS : !mpi.retval -> i1

// in order to check gainst other classes of errors, one must first call
// MPI_Error_class
%err_class = mpi.error_class %err : !mpi.retval -> !mpi.retval

// Check against specific error code
%is_err_rank = mpi.retval_check %err_class = MPI_ERR_RANK : !mpi.retval -> i1
```

*Note:*
 - We could also model `!mpi.retval` as `i32` if we wanted to. Although all the MPI error classes and codes are library dependent, so modelling it as int may not be that helpful anyways.

# Open Questions:

## Operation Naming

We make use of a pretty standard translation from MPI names to MLIR operation names and types, where the first `_` is replaced by `.` and everything is lowercased. That way `MPI_Comm_rank` becomes `mpi.comm_rank`. We also introduce some operations that are needed due to MLIR abstraction (e.g. `mpi.retval_check`). We could prefix them similar to how it's done in the LLVM dialect to become `mpi.mlir.retval_check`.

## Supporting more MPI Datatypes

The current version can support many kinds of memref layouts in arguments by mapping them to MPI strided datatypes.
MPI is able to express even more datatypes like heterogeneous arrays and structs. This is however not explored as part of this
RFC.
