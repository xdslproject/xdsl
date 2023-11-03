# MPI Dialect: RFC

This dialect models the Message Passing Interface (MPI), version 4.0. It is meant
to serve as a targetable dialect, that can be lowered to multiple MPI implementations
and hide differences in ABI. The dialect models the functions of the MPI
specification as close to 1:1 as possible while preserving SSA value semantics where it
makes sense, and uses `memref` types instead of bare pointers.

For an in-depth documentation of the MPI library interface, please refer to official documentation
such as the [OpenMPI online documentation](https://www.open-mpi.org/doc/current/).

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

Note that MPI has 7 different send modes, this RFC will only cover the two most commonly used (Blocking and Nonblocking).

# Blocking  Communication


```mlir
func.func @mpi_test(%ref : memref<100xf32>) -> () {
    mpi.init

    %rank = mpi.comm_rank : i32

    mpi.send (%ref, %rank, %tag) : memref<100xf32>, i32, i32

    mpi.recv (%ref, %rank, %tag) : memref<100xf32>, i32, i32

    mpi.finalize

    func.return
}
```

Here is the detailed operation definitions:
### `mpi.comm_rank` (mpi::CommRankOp)

_Get the current rank, equivalent to `MPI_Comm_rank(MPI_COMM_WORLD, &rank)`_.


Syntax:

```
operation ::= `mpi.comm_rank` attr-dict `:` type($result)
```

Communicators other than `MPI_COMM_WORLD` are not supprted for now.
Inspecting the functions return value (error code) is also not supported.

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | 32-bit signless integer


### `mpi.finalize` (mpi::FinalizeOp)

_Finalize the MPI library, equivalent to `MPI_Finalize()`_


Syntax:

```
operation ::= `mpi.finalize` attr-dict
```

This function cleans up the MPI state. Afterwards, no MPI methods may be invoked
(excpet for MPI_Get_version, MPI_Initialized, and MPI_Finalized).
Notably, MPI_Init cannot be called again in the same program.

Inspecting the functions return value (error code) is not supported.


### `mpi.init` (mpi::InitOp)

_Initialize the MPI library, equivalent to `MPI_Init(NULL, NULL)`_


Syntax:

```
operation ::= `mpi.init` attr-dict
```

This operation must preceed most MPI calls (except for very few exceptions,
please consult with the MPI specification on these).

Passing &argc, &argv is not supported currently.
Inspecting the functions return value (error code) is also not supported.


### `mpi.recv` (mpi::RecvOp)

_Equivalent to `MPI_Recv(ptr, size, dtype, dest, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE)`_


Syntax:

```
operation ::= `mpi.recv` attr-dict $ref `:` type($ref) `,` $tag `:` type($tag) `,` $rank `:` type($rank)
```

MPI_Recv performs a blocking receive of `size` elements of type `dtype` from rank `dest`.
The `tag` value and communicator enables the library to determine the matching of
multiple sends and receives between the same ranks.

Communicators other than `MPI_COMM_WORLD` are not supprted for now.
The MPI_Status is set to `MPI_STATUS_IGNORE`, as the status object is not yet ported to MLIR.
Inspecting the functions return value (error code) is also not supported.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `ref` | memref of any type values
| `tag` | 32-bit signless integer
| `rank` | 32-bit signless integer


### `mpi.send` (mpi::SendOp)

_Equivalent to `MPI_Send(ptr, size, dtype, dest, tag, MPI_COMM_WORLD)`_


Syntax:

```
operation ::= `mpi.send` attr-dict $ref `:` type($ref) `,` $tag `:` type($tag) `,` $rank `:` type($rank)
```

MPI_Send performs a blocking send of `size` elements of type `dtype` to rank `dest`.
The `tag` value and communicator enables the library to determine the matching of
multiple sends and receives between the same ranks.

Communicators other than `MPI_COMM_WORLD` are not supprted for now.
Inspecting the functions return value (error code) is also not supported.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `ref` | memref of any type values
| `tag` | 32-bit signless integer
| `rank` | 32-bit signless integer


# Nonblocking Communication

For nonblocking communication, a new datatype `!mpi.request`  is introduced. This is directly equivalent to the `MPI_Request` type defined by MPI.

Since MPI_Requests are mutable objects that are always passed by reference, we decide to model them inside memrefs and pass them as memref+index. This is consistent with how they are most often used in actual HPC programs (i.e. a stack-allocated array of `MPI_Request` objects).

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

# Collectives

TODO

# Handling `MPI_Status`

In order to handle MPI Status, we would introduce an optional result value of type `!mpi.status`. The `MPI_Status` is defined to be a struct with at least three fields (`MPI_SOURCE`, `MPI_TAG` and `MPI_ERROR`). Additionally, one can get the number of elements sent the from a status object using the `MPI_Get_count` function. We provide an accessor operation for these fields and additional operations for `MPI_Get_count`.

```mlir
%status = mpi.send (%ref, %rank, %tag) : (memref<100xf32>, i32, i32) -> !mpi.status

// access struct members:
%source = mpi.status_get_field %status[MPI_SOURCE] : !mpi.status -> i32
%tag = mpi.status_get_field %status[MPI_TAG] : !mpi.status -> i32
%err = mpi.status_get_field %status[MPI_ERROR] : !mpi.status -> !mpi.retval

%count = mpi.get_count %status : !mpi.status -> i32
```

# Lowering

This part gets into the ABI differences between implementation. We highly recommend the paper on [MPI Application Binary Interface Standardization](https://arxiv.org/pdf/2308.11214.pdf) as a primer for this section.

We have implemented an example showing off how we lower our initial patch to both MPICH and OpenMPI style ABIs (using xDSL for quick prototyping). We hope that the mess below is enough argument in favour of introducing the MPI dialect abstraction:

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

// Loweing to OpenMPI's opaque struct pointers:

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

We slightly prefer supporting to multiple libraries instead of an MLIR runtime, but don't want to rule out anything yet. The ABI standardisation efforts put forth by Hammond et al. hint at a more unified landscape in the future.

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
