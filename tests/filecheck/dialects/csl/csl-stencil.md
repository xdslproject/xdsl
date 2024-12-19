# Stencil to CSL lowering

Proposal: An intermediate `csl_stencil` dialect that manages communicating data across
the stencil pattern, sending to plus receiving from all neighbours.

Point in the pipeline: After applying `distribute-stencil`, `canonicalize`, and `stencil-tensorize-z-dim` transforms.
These passes transform the code into a representation that expresses that each PE holds one
vector of z-values, while accesses to  differing x/y coordinates require communicating with
neighbouring PEs.

The goal of `csl_stencil` is to more closely express on a high level what our low-level stencil (communications)
library does.

## Existing Lowering Pipeline

Our starting point may be a raw, untransformed stencil representation (before shape inference) such as follows: 

```
builtin.module {
  func.func @gauss_seidel_func(%a : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>, %b : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>) {
    %0 = "stencil.load"(%a) : (!stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>) -> !stencil.temp<?x?x?xf32>
    %1 = "stencil.apply"(%0) ({
    ^0(%2 : !stencil.temp<?x?x?xf32>):
      %3 = arith.constant 1.666600e-01 : f32
      %4 = "stencil.access"(%2) {"offset" = #stencil.index[1, 0, 0]} : (!stencil.temp<?x?x?xf32>) -> f32
      %5 = "stencil.access"(%2) {"offset" = #stencil.index[-1, 0, 0]} : (!stencil.temp<?x?x?xf32>) -> f32
      %6 = "stencil.access"(%2) {"offset" = #stencil.index[0, 0, 1]} : (!stencil.temp<?x?x?xf32>) -> f32
      %7 = "stencil.access"(%2) {"offset" = #stencil.index[0, 0, -1]} : (!stencil.temp<?x?x?xf32>) -> f32
      %8 = "stencil.access"(%2) {"offset" = #stencil.index[0, 1, 0]} : (!stencil.temp<?x?x?xf32>) -> f32
      %9 = "stencil.access"(%2) {"offset" = #stencil.index[0, -1, 0]} : (!stencil.temp<?x?x?xf32>) -> f32
      %10 = arith.addf %9, %8 : f32
      %11 = arith.addf %10, %7 : f32
      %12 = arith.addf %11, %6 : f32
      %13 = arith.addf %12, %5 : f32
      %14 = arith.addf %13, %4 : f32
      %15 = arith.mulf %14, %3 : f32
      "stencil.return"(%15) : (f32) -> ()
    }) : (!stencil.temp<?x?x?xf32>) -> !stencil.temp<?x?x?xf32>
    "stencil.store"(%1, %b) {"bounds" = #stencil.bounds[0, 0, 0] : [1022, 510, 510]} : (!stencil.temp<?x?x?xf32>, !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>) -> ()
    func.return
  }
}
```

To begin lowering, we can apply the following passes:
* `distribute-stencil{strategy=2d-grid slices=1022,510 restrict_domain=true}`
  * Distributes the stencil compute grid across the physical grid of compute nodes (or PEs).
    Assuming the default x-y split across x-y PEs, we can
    decompose the compute grid from (x,y,z) into (1,1,z) slices, each PE handling one batch
    of z-values. The return type of `stencil.apply` is changed to `!stencil.temp<[0,1]x[0,1]x[0,510]xf32>`
    (similarly for other ops).
  * Runs the `stencil-shape-inference` pass as a dependency.
  * As a distributed computation needs to exchange data, the pass inserts `dmp.swap` ops
    to indicate where this needs to happen. The semantics is to indicate that a bi-directional
    data exchange is needed between this node and a list of neighbouring nodes.
* `canonicalize`
  * (unverified) Removes redundant data transfers (`dmp.swap`) where the data has
    previously been exchanged and has not been invalidated by a `stencil.store`. 
* `stencil-tensorize-z-dimension`
  * Expresses the stencil computation as operating on tensors of z-values rather than scalar
    z-values.

The following result is slightly reformatted for readability:

```
builtin.module {
  func.func @gauss_seidel_func(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
    %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
    "dmp.swap"(%0) {"topo" = #dmp.topo<1022x510>, "swaps" = [
            #dmp.exchange<at [1, 0, 0] size [1, 1, 510] source offset [-1, 0, 0] to [1, 0, 0]>,
            #dmp.exchange<at [-1, 0, 0] size [1, 1, 510] source offset [1, 0, 0] to [-1, 0, 0]>,
            #dmp.exchange<at [0, 1, 0] size [1, 1, 510] source offset [0, -1, 0] to [0, 1, 0]>,
            #dmp.exchange<at [0, -1, 0] size [1, 1, 510] source offset [0, 1, 0] to [0, -1, 0]>
        ]} : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> ()
    %1 = stencil.apply(%2 = %0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) {
      %3 = arith.constant 1.666600e-01 : f32
      %4 = stencil.access %2[1, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %6 = stencil.access %2[-1, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %8 = stencil.access %2[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %10 = stencil.access %2[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %12 = stencil.access %2[0, 1] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %14 = stencil.access %2[0, -1] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %5 = "tensor.extract_slice"(%4) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %7 = "tensor.extract_slice"(%6) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %9 = "tensor.extract_slice"(%8) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %11 = "tensor.extract_slice"(%10) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %13 = "tensor.extract_slice"(%12) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %15 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %16 = arith.addf %15, %13 : tensor<510xf32>
      %17 = arith.addf %16, %11 : tensor<510xf32>
      %18 = arith.addf %17, %9 : tensor<510xf32>
      %19 = arith.addf %18, %7 : tensor<510xf32>
      %20 = arith.addf %19, %5 : tensor<510xf32>
      %21 = tensor.empty() : tensor<510xf32>
      %22 = linalg.fill ins(%3 : f32) outs(%21 : tensor<510xf32>) -> tensor<510xf32>
      %23 = arith.mulf %20, %22 : tensor<510xf32>
      stencil.return %23 : tensor<510xf32>
    }
    stencil.store %1 to %b ([0, 0] : [1, 1]) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }
}
```

Note: `dmp.swap` has not been touched by `stencil-tensorize-z-dim` and remains 3-dimensional, while
the remainder of the stencil code 2-dimensional after the transformation (requesting feedback).

# Strategy

The `dmp.swap` ops describe the data movement required, expressing a point-to-point send and receive
exchange between us and a specified neighbour. Unfortunately, this cannot be translated
directly on CS-2, as the router has limited configurations. The currently chosen communication
strategy enables one-shot distribution of data across a fixed stencil shape. This might occasionally
lead to redundant transfers in cases where specific data points are not needed for the computation -
however, as such stencils on cs-2 are reported to be compute-bound, this may have limited impact.

We propose `csl_stencil.prefetch` as an op that communicates a given buffer across the stencil shape,
as a pure data transfer op without performing any compute.
As an example, where `dmp.swap` indicates that one or several buffers need to be swapped before a
`stencil.apply`, a simple lowering strategy would be to prefetch all buffers before proceeding with
the compute.

Prefetching may have a potentially significant memory overhead, as one buffer per neighbour
needs to be retained, for all prefetched buffers and for all neighbours whose data we need.
Ideally, we would prefer to immediately consume the data upon receiving it, reducing it to a
single buffer. The op that reduces data from many neighbours to a single thing is `stencil.apply`.
We therefore propose wherever possible to further lower `csl_stencil.prefetch` together with
`stencil.apply` into an op that combines the functionality of both: `csl_stencil.apply`.

As the name suggests, `csl_stencil.apply` combines the data transfer of a `csl_stencil.prefetch`
(built from a `dmp.swap`) with the stencil computation of `stencil.apply`, and should be built
from both ops combined.

To summarize, we propose the following ops:
* `csl_stencil.prefetch` - pure data transfer across the stencil pattern, built from `dmp.swap`
* `csl_stencil.apply` - transfers and consumes buffers across the stencil pattern, built from `csl_stencil.prefetch` and `stencil.apply`
* `csl_stencil.access` - behaves like a `stencil.access` but to a prefetched buffer

One possible optimisation strategy at this level of abstraction is to optimise for space,
which could be achieved by lowering as many `csl_stencil.prefetch` ops as possible to `csl_stencil.apply` ops.
This would reduce the need for intermediate buffers, and could be achieved by a restructuring of
`stencil.apply` ops applied beforehand.

## Step 1: Buffer prefetching

In this step, `dmp.swap` is lowered to `csl_stencil.prefetch`. The op returns a `tensor<4x510xf32>`,
indicating that we have received data from 4 neighbours (any redundant exchange can be omitted here).
The `tensor` is passed as an additional argument to `stencil.apply` and is accessed via `csl_stencil.access`.
This op is like `stencil.access`, but operates on prefetched buffers and does not require an offset,
as ghost cells have not been communicated. This also means that some `tensor.extract_slice` ops
can be dropped, as they are no longer needed to remove any ghost cells.

The `stencil.access` ops are retained for accesses to data held by this node.

The resulting mlir is as follows:

```
builtin.module {
  func.func @gauss_seidel_func(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
    %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
    %pref = csl_stencil.prefetch(%0) {"topo" = #dmp.topo<1022x510>, "swaps" = [
            #csl_stencil.exchange<to [1, 0]>,
            #csl_stencil.exchange<to [-1, 0]>,
            #csl_stencil.exchange<to [0, 1]>,
            #csl_stencil.exchange<to [0, -1]>
        ]} : (!stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>) -> (tensor<4x510xf32>)
    %1 = stencil.apply(%2 = %0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %3 = %pref : tensor<4x255xf32>) -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) {
      %4 = arith.constant 1.666600e-01 : f32
      %5 = csl_stencil.access %3[1, 0] : tensor<4x255xf32>
      %6 = csl_stencil.access %3[-1, 0] : tensor<4x255xf32>
      %7 = stencil.access %2[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %8 = stencil.access %2[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
      %9 = "tensor.extract_slice"(%7) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %10 = "tensor.extract_slice"(%8) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %11 = csl_stencil.access %3[0, 1] : tensor<4x255xf32>
      %12 = csl_stencil.access %3[0, -1] : tensor<4x255xf32>
      %13 = arith.addf %12, %11 : tensor<510xf32>
      %14 = arith.addf %13, %10 : tensor<510xf32>
      %15 = arith.addf %14, %9 : tensor<510xf32>
      %16 = arith.addf %15, %6 : tensor<510xf32>
      %17 = arith.addf %16, %5 : tensor<510xf32>
      %18 = tensor.empty() : tensor<510xf32>
      %19 = linalg.fill ins(%4 : f32) outs(%18 : tensor<510xf32>) -> tensor<510xf32>
      %20 = arith.mulf %17, %19 : tensor<510xf32>
      stencil.return %20 : tensor<510xf32>
    }
    stencil.store %1 to %b ([0, 0] : [1, 1]) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }
}
```

## Interlude: The CSL stencil comms library targeted

This is a basic example of what the proposed dialect aims to accomplish. The actual underlying
CSL library is slightly more complex, specifically in two ways:
- First, it allows splitting communication into chunks. Splitting the communication in to fixed-size
chunks effectively fixes the space requirement of communication buffers, and becomes useful when
dealing with large numbers of z-values.
- Second, communication is asynchronous and needs to trigger a callback when completed.
The above code assumes synchronous communication and compute, and will (in the second step below)
need to be transformed to perform asynchronous communications.

The CSL stencil communications library provides callbacks for both of these cases. Recall the
function definition and parameter description:


```
// communicates the content of send_dsd to all points in the stencil, and receives data from all points in the stencil
// communication happens in a specified number of chunks at a fixed chunk size (module parameter)
// this function takes the following arguments:
//   - send_dsd:           data to be sent, must be at least of length chunkSize
//   - num_chunks:         the number of chunks to be sent
//   - clear_recv_buf_cb:  callback invoked after receiving one chunk of data from each point in the stencil.
//                         takes one arg 'offset' to indicate that the received data chunks are positioned
//                         at [offset, offset+chunkSize] in the sender's send_dsd.
//                         to access data in the receive buffers, use getRecvBufDsd(), getRecvBufDsdByDir(), or getRecvBufDsdByNeighbor().
//                         the total number of invocations of the callback equals numChunks amount of time.
//   - communicate_cb:     callback invoked once after all communication (sending and receiving) is completed.

fn communicate(send_dsd: mem1d_dsd,
               num_chunks: i16,
               clear_recv_buf_cb: *const fn(i16)void,
               communicate_cb: *const fn()void
              ) void { ... }
```

## Step 2: Communication with compute

The above stencil comms CSL function is invoked with two callbacks with distinct tasks, and
it's best to understand them from their types:
* The `clear_recv_buf_cb` takes a collection of partial z-value tensors and reduces them to one
  partial z-value tensor. It is invoked one for each communicated slice (essentially acting like a loop body),
  and builds up one combined full (non-partial) z-value tensor.
  * Note: Both `csl_stencil.prefetch` and `csl_stencil.apply` are lowered to invoke the same (above)
    function with `clear_recv_buf_cb` callback - the difference is that prefetch copies the data
    received from `n` neighbours into `n` buffers, while `csl_stencil.apply` consumes the data
    received from `n` neighbours into 1 buffer by applying the compute given in the `stencil.apply` body.
* The `communicate_cb` takes the reduced data and/or that of any prefetched buffers, applying
  any computation that can only be done after communication has finished. It also performs
  the semantic task of `stencil.return`.

The following shows `csl_stencil.apply`, combining the functionality of `csl_stencil.prefetch`
and `stencil.apply`, with two code blocks corresponding to the two callbacks (in order):

```
builtin.module {
  func.func @gauss_seidel_func(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
    %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>

    %1 = tensor.empty() : tensor<510xf32>
    %2 = csl_stencil.apply(%3 = %0 : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>, %empty_res = %1 : tensor<510xf32>) {"topo" = #dmp.topo<1022x510>, "swaps" = [
            #csl_stencil.exchange<to [1, 0]>,
            #csl_stencil.exchange<to [-1, 0]>,
            #csl_stencil.exchange<to [0, 1]>,
            #csl_stencil.exchange<to [0, -1]>
        ]} -> (!stencil.temp<[0,1]x[0,1]xtensor<510xf32>>) ({
      ^0(%recv : !stencil.temp<[-1,2]x[-1,2]xtensor<255xf32>>, %offset : i64, %iter_arg : tensor<510xf32>):
        // reduces chunks from neighbours into one chunk (clear_recv_buf_cb)
        %4 = csl_stencil.access %recv[1, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<255xf32>>
        %5 = csl_stencil.access %recv[-1, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<255xf32>>
        %6 = csl_stencil.access %recv[0, 1] : !stencil.temp<[-1,2]x[-1,2]xtensor<255xf32>>
        %7 = csl_stencil.access %recv[0, -1] : !stencil.temp<[-1,2]x[-1,2]xtensor<255xf32>>

        %8 = arith.addf %4, %5 : tensor<255xf32>
        %9 = arith.addf %8, %6 : tensor<255xf32>
        %10 = arith.addf %9, %7 : tensor<255xf32>

        %11 = "tensor.insert_slice"(%10, %iter_arg, %offset) : (tensor<255xf32>, tensor<510xf32>, i64) -> tensor<510xf32>
        csl_stencil.yield %11 : tensor<510xf32>
      }, {
      ^0(%rcv : tensor<510xf32>):
        // takes combined chunks and applies further compute (communicate_cb)
        %12 = stencil.access %3[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
        %13 = stencil.access %3[0, 0] : !stencil.temp<[-1,2]x[-1,2]xtensor<512xf32>>
        %14 = "tensor.extract_slice"(%12) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
        %15 = "tensor.extract_slice"(%13) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>

        %16 = arith.addf %rcv, %14
        %17 = arith.addf %16, %15

        %18 = arith.constant 1.666600e-01 : f32
        %19 = tensor.empty() : tensor<510xf32>
        %20 = linalg.fill ins(%18 : f32) outs(%19 : tensor<510xf32>) -> tensor<510xf32>
        %21 = arith.mulf %17, %20 : tensor<510xf32>

        csl_stencil.return %21 : tensor<510xf32>
      })
    
    stencil.store %2 to %b ([0, 0] : [1, 1]) : !stencil.temp<[0,1]x[0,1]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }
}
```

The first code block acts effectively like a loop body over the communicated chunks, and is
executed once for each communicated chunk. It is initialised
with an empty tensor (`%1`) which it builds up in `%iter_args`, inspired by the argument
to `scf.for` of the same name.

Please note: Some of the `arith` ops have been manually re-ordered before being moved
into the two callback blocks.