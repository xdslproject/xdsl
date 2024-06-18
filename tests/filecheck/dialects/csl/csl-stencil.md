# Stencil to CSL lowering

Proposal: A `csl_stencil` dialect that manages communicating data across
the stencil pattern, sending to plus receiving from all neighbours.

Starting point: Output of the `stencil-tensorize-z-dim` transform. This representation
expresses that each PE holds one vector of z-values, while accesses to differing x/y coordinates
require communicating with neighbouring PEs.

The goal is to more closely express on a high level what our low-level stencil (communications)
library does.

## Starting point 
Recall the following starting representation of tensorized z-dim values - please note, the
`tensor.extract_slice` ops manage the halo and have been slightly re-ordered for improved readability: 

``` mlir
builtin.module {
  func.func @gauss_seidel(%a : memref<1024x512xtensor<512xf32>>, %b : memref<1024x512xtensor<512xf32>>) {
    %0 = stencil.external_load %a : memref<1024x512xtensor<512xf32>> -> !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    %1 = stencil.load %0 : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
    %2 = stencil.external_load %b : memref<1024x512xtensor<512xf32>> -> !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    %3 = stencil.apply(%4 = %1 : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>) -> (!stencil.temp<[0,1022]x[0,510]xtensor<510xf32>>) {
      %5 = arith.constant 1.666600e-01 : f32
      %6 = stencil.access %4[1, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
      %8 = stencil.access %4[-1, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
      %10 = stencil.access %4[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
      %12 = stencil.access %4[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
      %14 = stencil.access %4[0, 1] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
      %16 = stencil.access %4[0, -1] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
      %7 = "tensor.extract_slice"(%6) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %9 = "tensor.extract_slice"(%8) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %11 = "tensor.extract_slice"(%10) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %13 = "tensor.extract_slice"(%12) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %15 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %17 = "tensor.extract_slice"(%16) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %18 = arith.addf %17, %15 : tensor<510xf32>
      %19 = arith.addf %18, %13 : tensor<510xf32>
      %20 = arith.addf %19, %11 : tensor<510xf32>
      %21 = arith.addf %20, %9 : tensor<510xf32>
      %22 = arith.addf %21, %7 : tensor<510xf32>
      %23 = tensor.empty() : tensor<510xf32>
      %24 = linalg.fill ins(%5 : f32) outs(%23 : tensor<510xf32>) -> tensor<510xf32>
      %25 = arith.mulf %22, %24 : tensor<510xf32>
      stencil.return %25 : tensor<510xf32>
    }
    stencil.store %3 to %2 ([0, 0] : [1022, 510]) : !stencil.temp<[0,1022]x[0,510]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }
}
```

## Step 1: A simple, synchronous csl-stencil library

This document shows a series of small transformations, here applied manually. These may be combined
into one larger transformation when implemented, but are shown as a sequence of small transforms
for the purposes of illustration.

The initial step introduces a (simplified) version of the `csl_stencil.communicate` function.
Its semantics are that it takes 'this PEs' z-values and broadcasts them across the neighbours
within the stencil pattern, and also receives their z-values. Since in this example, there are
4 neighbours in the stencil, the function returns `memref<4xtensor<510xf32>>`, 4 being the number
of neighbours.

Subsequently, the received data is accessed by invoking `csl_stencil.access`, which is inspired by
`stencil.access`. The remaining parts of the computation are left unchanged. Importantly,
accessing 'this PEs' z-values is also left unchanged and remain `stencil.access` ops:

```
builtin.module {
  func.func @gauss_seidel(%a : memref<1024x512xtensor<512xf32>>, %b : memref<1024x512xtensor<512xf32>>) {
    %0 = stencil.external_load %a : memref<1024x512xtensor<512xf32>> -> !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    %1 = stencil.load %0 : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
    %2 = stencil.external_load %b : memref<1024x512xtensor<512xf32>> -> !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    %3 = stencil.apply(%4 = %1 : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>) -> (!stencil.temp<[0,1022]x[0,510]xtensor<510xf32>>) {
      %5 = arith.constant 1.666600e-01 : f32

      // get this PEs z-values
      // we can communicate with or without including the halo, here it's without
      %6 = "tensor.extract_slice"(%4) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>

      // broadcasts this PE's z-values across the stencil pattern and receives z-values from other PE's in the stencil pattern
      %7 = csl_stencil.communicate(%6) : (tensor<510xf32>) -> (memref<4xtensor<510xf32>>)

      // after communicate is done, access other z-values and continue computation
      %8 = csl_stencil.access %7[1, 0] : (memref<4xtensor<510xf32>>) -> (tensor<510xf32>)
      %9 = csl_stencil.access %7[-1, 0] : (memref<4xtensor<510xf32>>) -> (tensor<510xf32>)
      %10 = csl_stencil.access %7[0, 1] : (memref<4xtensor<510xf32>>) -> (tensor<510xf32>)
      %11 = csl_stencil.access %7[0, -1] : (memref<4xtensor<510xf32>>) -> (tensor<510xf32>)

      // access to this PEs own z-values is not modified in this transform
      %12 = stencil.access %4[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
      %13 = "tensor.extract_slice"(%12) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %14 = stencil.access %4[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
      %15 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>

      %18 = arith.addf %11, %10 : tensor<510xf32>
      %19 = arith.addf %18, %15 : tensor<510xf32>
      %20 = arith.addf %19, %13 : tensor<510xf32>
      %21 = arith.addf %20, %9 : tensor<510xf32>
      %22 = arith.addf %21, %8 : tensor<510xf32>
      %23 = tensor.empty() : tensor<510xf32>
      %24 = linalg.fill ins(%5 : f32) outs(%23 : tensor<510xf32>) -> tensor<510xf32>
      %25 = arith.mulf %22, %24 : tensor<510xf32>
      stencil.return %25 : tensor<510xf32>
    }
    stencil.store %3 to %2 ([0, 0] : [1022, 510]) : !stencil.temp<[0,1022]x[0,510]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }
}
```

## Interlude: The CSL library we are targeting

This is a basic example of what the proposed dialect aims to accomplish. The actual underlying
CSL library is slightly more complex, specifically in two ways:
- First, it allows splitting communication into chunks. Splitting the communication in to fixed-size
chunks effectively fixes the space requirement of communication buffers, and becomes useful when
dealing with large numbers of z-values. It is unclear whether it allows for better interleaving
of communication and compute. 
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

## Step 2: Communicating in chunks

The next transformation step shows how the communication may be split into chunks. The
`csl_stencil.communicate` function here is passed a code block whose task is to process
one received chunk of data. This should eventually be lowered to the `clear_recv_buf_cb` callback.
The block is effectively like a loop body, invoked subsequently for each chunk. 
After receiving several chunks of data, the function needs to return something combining these chunks.
There are several ways of expressing this, and I don't mind which one to take. Here, an empty
tensor is passed in, acting as a loop carry variable, updated by each invocation of the block.

I have made the following simplifications:
- For simplicity, I assume that the block can return one combined `tensor<510xf32>`. Alternatively,
it may need to return `memref<4xtensor<510xf32>>`
- For simplicity, the `arith.addf` functions have been re-ordered


```
builtin.module {
  func.func @gauss_seidel(%a : memref<1024x512xtensor<512xf32>>, %b : memref<1024x512xtensor<512xf32>>) {
    %0 = stencil.external_load %a : memref<1024x512xtensor<512xf32>> -> !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    %1 = stencil.load %0 : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
    %2 = stencil.external_load %b : memref<1024x512xtensor<512xf32>> -> !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    %3 = stencil.apply(%4 = %1 : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>) -> (!stencil.temp<[0,1022]x[0,510]xtensor<510xf32>>) {
      %5 = arith.constant 1.666600e-01 : f32

      // get this PEs z-values
      // we can communicate with or without including the halo, here it's without
      %6 = "tensor.extract_slice"(%4) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>

      // the `communicate` function is invoked with property `num_chunks = 2` indicating that communication is done in 2 steps
      // the block arg processes each chunk of data before communication can continue
      // the result is built up in a variable inspired by `iter_arg` of `scf.for`

      %empty_res = tensor.empty() : tensor<510xf32>
      %7 = csl_stencil.communicate(%6, %empty_res) ({
      ^0(%recv : memref<4xtensor<255xf32>>, %offset : i64, %iter_arg : tensor<510xf32>):

        %8 = csl_stencil.access %recv[1, 0] : (memref<4xtensor<255xf32>>) -> (tensor<255xf32>)
        %9 = csl_stencil.access %recv[-1, 0] : (memref<4xtensor<255xf32>>) -> (tensor<255xf32>)
        %10 = csl_stencil.access %recv[0, 1] : (memref<4xtensor<255xf32>>) -> (tensor<255xf32>)
        %11 = csl_stencil.access %recv[0, -1] : (memref<4xtensor<255xf32>>) -> (tensor<255xf32>)

        // the goal here is not to do heavy compute, but to clear the receive-buffers, such that communication can continue

        %12 = arith.addf %9, %8 : tensor<255xf32>
        %13 = arith.addf %12, %10 : tensor<255xf32>
        %14 = arith.addf %13, %12 : tensor<255xf32>

        %inserted_slice = "tensor.insert_slice"(%14, %iter_arg, %offset) : (tensor<255xf32>, tensor<510xf32>, i64) -> tensor<510xf32>
        csl_stencil.yield %inserted_slice : tensor<510xf32>

      )} <{"num_chunks" = 2 : i16}> : (tensor<510xf32>, tensor<510xf32>) -> (tensor<510xf32>)

      %12 = stencil.access %4[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
      %13 = "tensor.extract_slice"(%12) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
      %14 = stencil.access %4[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
      %15 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>

      %16 = arith.addf %7, %13 : tensor<510xf32>
      %17 = arith.addf %6, %15 : tensor<510xf32>

      %23 = tensor.empty() : tensor<510xf32>
      %24 = linalg.fill ins(%5 : f32) outs(%23 : tensor<510xf32>) -> tensor<510xf32>
      %25 = arith.mulf %17, %24 : tensor<510xf32>
      stencil.return %25 : tensor<510xf32>
    }
    stencil.store %3 to %2 ([0, 0] : [1022, 510]) : !stencil.temp<[0,1022]x[0,510]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }
}
```

## Step 3: Communicating asynchronously via a callback

Finally, `csl_stencil.communicate` performs an asynchronous send and receive. When all chunks
of data are sent and received, the `communicate_cb` callback is invoked. (NB: Technically, this may not be needed but makes the
API nicer - the alternative would be that the caller has to keep count of how many times the
`clear_recv_buf_cb` has been invoked.)
This callback triggers the remaining computation, which we model by moving the remaining
compute ops into the code block. Note, that `csl_stencil.communicate` still returns a value
which is trivially returned by `stencil.apply`.

```
builtin.module {
  func.func @gauss_seidel(%a : memref<1024x512xtensor<512xf32>>, %b : memref<1024x512xtensor<512xf32>>) {
    %0 = stencil.external_load %a : memref<1024x512xtensor<512xf32>> -> !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    %1 = stencil.load %0 : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
    %2 = stencil.external_load %b : memref<1024x512xtensor<512xf32>> -> !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    %3 = stencil.apply(%4 = %1 : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>) -> (!stencil.temp<[0,1022]x[0,510]xtensor<510xf32>>) {
      %5 = arith.constant 1.666600e-01 : f32

      // get this PEs z-values
      // we can communicate with or without including the halo, here it's without
      %6 = "tensor.extract_slice"(%4) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>

      // the `communicate` function is invoked with property `num_chunks = 2` indicating that communication is done in 2 steps
      // the block arg processes each chunk of data before communication can continue
      // the result is built up in a variable inspired by `iter_arg` of `scf.for`

      %empty_res = tensor.empty() : tensor<510xf32>
      %7 = csl_stencil.communicate(%6, %empty_res) ({
      ^0(%recv : memref<4xtensor<255xf32>>, %offset : i64, %iter_arg : tensor<510xf32>):

        %8 = csl_stencil.access %recv[1, 0] : (memref<4xtensor<255xf32>>) -> (tensor<255xf32>)
        %9 = csl_stencil.access %recv[-1, 0] : (memref<4xtensor<255xf32>>) -> (tensor<255xf32>)
        %10 = csl_stencil.access %recv[0, 1] : (memref<4xtensor<255xf32>>) -> (tensor<255xf32>)
        %11 = csl_stencil.access %recv[0, -1] : (memref<4xtensor<255xf32>>) -> (tensor<255xf32>)

        // the goal here is not to do heavy compute, but to clear the receive-buffers, such that communication can continue

        %12 = arith.addf %9, %8 : tensor<255xf32>
        %13 = arith.addf %12, %10 : tensor<255xf32>
        %14 = arith.addf %13, %12 : tensor<255xf32>

        %inserted_slice = "tensor.insert_slice"(%14, %iter_arg, %offset) : (tensor<255xf32>, tensor<510xf32>, i64) -> tensor<510xf32>
        csl_stencil.yield %inserted_slice : tensor<510xf32>
      },{
      ^0(%res : tensor<510xf32>):

        %12 = stencil.access %4[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
        %13 = "tensor.extract_slice"(%12) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
        %14 = stencil.access %4[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
        %15 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>

        %16 = arith.addf %res, %13 : tensor<510xf32>
        %17 = arith.addf %6, %15 : tensor<510xf32>

        %23 = tensor.empty() : tensor<510xf32>
        %24 = linalg.fill ins(%5 : f32) outs(%23 : tensor<510xf32>) -> tensor<510xf32>
        %25 = arith.mulf %17, %24 : tensor<510xf32>

        csl_stencil.yield %25 : : tensor<510xf32>

      }) <{"num_chunks" = 2 : i16}> : (tensor<510xf32>, tensor<510xf32>) -> (tensor<510xf32>)

      stencil.return %7 : tensor<510xf32>
    })
    stencil.store %3 to %2 ([0, 0] : [1022, 510]) : !stencil.temp<[0,1022]x[0,510]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
    func.return
  }
}
```