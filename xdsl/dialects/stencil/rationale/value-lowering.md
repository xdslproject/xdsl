# Value-semantic stencil lowering

## Current design

Currently, the stencil is in some kind of mixed-semantics state.\
Let's have a look at a typical simple lowering:

```mlir
// Load input
%ut = stencil.load %u : !stencil.field<[-4,68]xf64> -> !stencil.temp<[-1,65]xf64>

// Value-semantics compute
%vt = stencil.apply(%uarg = %ut : !stencil.temp<[-1,65]xf64>) -> (!stencil.temp<[0,64]xf64>) {

    %left = stencil.access %uarg[-1] : !stencil.temp<[-1,65]xf64>
    %center = stencil.access %uarg[0] : !stencil.temp<[-1,65]xf64>
    %right = stencil.access %uarg[1] : !stencil.temp<[-1,65]xf64>

    %value = func.call @compute(%left, %center, %right) : (f64, f64, f64) -> f64

    stencil.return %value : f64
}

// Store outputs
stencil.store %vt to %v ([0] : [64]) : !stencil.temp<[0,64]xf64> to  !stencil.field<[-4,68]xf64>
```

The current lowering lowers everything from this state to loops over memref directly:

```mlir
    %v_storeview = "memref.subview"(%v) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 64>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72xf64>) -> memref<64xf64, strided<[1], offset: 4>>

    %u_loadview = "memref.subview"(%u) <{"static_offsets" = array<i64: 4>, "static_sizes" = array<i64: 66>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<72xf64>) -> memref<66xf64, strided<[1], offset: 4>>

    %0 = arith.constant 0 : index
    %1 = arith.constant 1 : index
    %2 = arith.constant 64 : index
    "scf.parallel"(%0, %2, %1) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
    ^0(%3 : index):
    
      %left = arith.constant -1 : index
      %left_1 = arith.addi %3, %left : index
      %left_2 = memref.load %u_loadview[%left_1] : memref<66xf64, strided<[1], offset: 4>>
      %center = memref.load %u_loadview[%3] : memref<66xf64, strided<[1], offset: 4>>
      %right = arith.constant 1 : index
      %right_1 = arith.addi %3, %right : index
      %right_2 = memref.load %u_loadview[%right_1] : memref<66xf64, strided<[1], offset: 4>>

      %value = func.call @compute(%left_2, %center, %right_2) : (f64, f64, f64) -> f64

      memref.store %value, %v_storeview[%3] : memref<64xf64, strided<[1], offset: 4>>
      scf.yield
    }) : (index, index, index) -> ()
```

The OEC dialect restrict things so that a field can only be loaded or stored too, *never both* in a same program.

As discussed in [value-semantics.md](), this trivially blocks some more value-semantic style expressions using the dialect.

Why this restriction? It just simplifies this one-shot translation.

Let's look at another variant of the limitation to start working on a simpler problem than value-semantic iteration:
```mlir
// Load input
%ut = stencil.load %u : !stencil.field<[-4,68]xf64> -> !stencil.temp<[0,64]xf64>

// Value-semantics compute
%vt = stencil.apply(%uarg = %ut : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[0,64]xf64>) {
    %center = stencil.access %uarg[0] : !stencil.temp<[0,64]xf64>
    %value = func.call @compute(%center) : (f64) -> f64
    stencil.return %value : f64
}

// Store outputs
// Cannot load and store to the same field !
stencil.store %vt to %u ([0] : [64]) : !stencil.temp<[0,64]xf64> to  !stencil.field<[-4,68]xf64>
```

One can see that this would be okay to lower as-is, as there are no "space dependencies" in the stencil at hand: it's an element-wise mapping.\
But this is loading and storing from a same field!

## A case for bufferization

Now, those objects and this problem are very similar to MLIR's `memref`, `tensor`, and `linalg`! There is friction with using `linalg`, I'm happy to come back to it, but let's avoid the bikeshed first then.\
Even with `linalg` specifically out of the landscape, this is exactly the problem called "bufferization" in MLIR. And if we do not want to consider using the existing infrastructure, I think we can at least consider its design?

> Skipping over the details, sensible bufferization from inputs to outputs would require destination-passing-style. I don't think it is necessary for the stencil dialet, so let's try a bufferization variant, in the stencil mindset: from outputs to inputs.

As usual, part of the rationale is reducing the problem to more local occurences, and apply a more progressive transformation from one state to another. Instead of lowering everything at once, we could allow `stencil.apply` to optionally work on `stencil.field`s directly, and have a progressive assignments of `field`s to `temp`s:


## First notes

MLIR is using destination-passing style to simplify this process. I'm still trying to avoid this for the stencil dialect, it makes everything feels bloated and somehow less elegant at the high-level.
MLIR also has two core operations to glue different bufferization stages : bufferization.to_memref and bufferization.to_tensor. Their respective signatures are `tensor -> memref` and `tensor -> memref`, and act as an unrealized cast specialized for bufferization.

So, if fields are memrefs and temps are tensors, we want (in the absence of just interfacing away memref and tensor :) ) the equivalents `temp -> field` and `field -> temp`. Wait, we have `field -> temp`! It's just `stencil.load`!

`temp -> field` is reminding of `stencil.store`, but it is `(temp, field) -> ()`. It does not say "give me the buffer containing those values" but "Put those values in that known buffer" (The closest `bufferization` equivalent is `materialize_in_destination`). I tried a lot of things, but it just makes everything harder (i.e, unecessarily complex!) to not use that approach.

We have a weird beast in the stencil dialect already : `stencil.buffer`, being a `temp -> temp`, and meaning "Don't worry, an intermediary buffer will exist for these values". I want now to explore adding a `stencil.alloc` (to not overload buffer too much) explicitely allocating a field, and make `stencil.buffer` able to go from `temp -> temp` to `temp -> field`, i.e;, mimicing `bufferization.to_memref`.

### A stencil.apply's result is allocated:

We replace *all* value-semantics results of an apply by an allocation, compute directly on it, and load it for next ops:

```mlir
// Load input
%ut = stencil.load %u : !stencil.field<[-4,68]xf64> -> !stencil.temp<[0,64]xf64>

%vt_b = stencil.alloc() : !stencil.field<[0,64]xf64>
stencil.apply(%uarg = %ut : !stencil.temp<[0,64]xf64>) -> (%vt_b : !stencil.field<[0,64]xf64>) {
    %center = stencil.access %uarg[0] : !stencil.temp<[0,64]xf64>
    %value = func.call @compute(%center) : (f64) -> f64
    stencil.return %value : f64
}
%vt = stencil.load %vt_b : !stencil.field<[0,64]xf64> -> !stencil.temp<[0,64]xf64>

stencil.store %vt to %v : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
```

### load-stores are just backpropagating the final buffer if safe and the loaded one is not used anymore.

We here match on the `%vt` situation: we have a load *being the last use of its buffer*, and a store to an existing buffer `%v`, that is *unused between since the loaded buffer is defined*.

We can thus replace the destination:
```mlir
// Load input
%ut = stencil.load %u : !stencil.field<[-4,68]xf64> -> !stencil.temp<[0,64]xf64>

%vt_b = stencil.alloc() : !stencil.field<[0,64]xf64>
stencil.apply(%uarg = %ut : !stencil.temp<[0,64]xf64>) -> (%v : !stencil.field<[-4,68]xf64>) {
    %center = stencil.access %uarg[0] : !stencil.temp<[0,64]xf64>
    %value = func.call @compute(%center) : (f64) -> f64
    stencil.return %value : f64
}
```

And we can simply remove unused allocations.

### A stencil apply's operand is just replaced by its underlying buffer
```mlir
// Load input
%ut = stencil.load %u : !stencil.field<[-4,68]xf64> -> !stencil.temp<[0,64]xf64>
%ut_b = stencil.buffer(%ut) : !stencil.temp<[0,64]xf64> -> !stencil.field<[0,64]xf64>

stencil.apply(%uarg = %ut_b : !stencil.field<[0,64]xf64>) -> (%v : !stencil.field<[-4,68]xf64>) {
    %center = stencil.access %uarg[0] : !stencil.temp<[0,64]xf64>
    %value = func.call @compute(%center) : (f64) -> f64
    stencil.return %value : f64
}
```

### The underlying buffer of a loaded field is the field

If the field is not modified between the `stencil.load` and the last use of the underlying buffer, inclusive.

```mlir
stencil.apply(%uarg = %u : !stencil.field<[-4,68]xf64>) -> (%v : !stencil.field<[-4,68]xf64>) {
    %center = stencil.access %uarg[0] : !stencil.temp<[0,64]xf64>
    %value = func.call @compute(%center) : (f64) -> f64
    stencil.return %valuex : f64
}
```

i.e, if we would have the same output buffer: 

```mlir
// Load input
%ut = stencil.load %u : !stencil.field<[-4,68]xf64> -> !stencil.temp<[0,64]xf64>
%ut_b = stencil.buffer(%ut) : !stencil.temp<[0,64]xf64> -> !stencil.field<[0,64]xf64>

stencil.apply(%uarg = %ut_b : !stencil.field<[0,64]xf64>) -> (!stencil.temp<[0,64]xf64>, %u : !stencil.field<[-4,68]xf64>) {
    %center = stencil.access %uarg[0] : !stencil.temp<[0,64]xf64>
    %value = func.call @compute(%center) : (f64) -> f64
    stencil.return %value, %value : f64
}
```

The general answer would be to make a copy, as the buffer is used at this point. We can special-case if the only conflict is an in-place stencil like above though.

The stencil is now in full reference-semantic state: all applys are reading fro, and writing to *buffers*, making the lowering simpler - as well as any reasonning about stencils and buffers!

## Having a look at loops

### Minimal example

We now have a satisfactory bufferization transformation for the stencil dialect itself.

What about loops? Are we extending everything to loops and every next mixed abstraction? I want to explore the idea of allowing bufferization boundaries here too, and using MLIR's bufferization on existing dialect, that is, lowering any remaining temp to tensor and field to memref, and let MLIR bufferize loops, functions, as it knows how to already.

```mlir
%ut = stencil.load(%u) : !stencil.field<[-4, 68]xf64> -> !stencil.temp<[0,64]xf64>
%uout = scf.for %time = %0 to %T step %1 iter_args(%ui = %ut) -> (!stencil.temp<[0,64]xf64>) {
    %unew = stencil.apply(%uarg = %ui : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[0,64]xf64>) {
        %v = stencil.access %uarg[0] : !stencil.temp<[0,64]xf64>
        %vnew = func.call @compute(%v) : (f64) -> f64
        stencil.return %vnews
    }
    scf.yield(%unew)
}
stencil.store(%uout, %u) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
```

Let's apply a similar rationale to scf.for than stencil.apply: we want it to use fields instead of temps, operands just use buffer, results get replaced by allocated operands:
```mlir
%ut = stencil.load(%u) : !stencil.field<[-4, 68]xf64> -> !stencil.temp<[0,64]xf64>
%ut_b = stencil.buffer(%ut) : !stencil.temp<[0,64]xf64> -> !stencil.field<[0, 64]xf64>
%uout_b = scf.for %time = %0 to %T step %1 iter_args(%ui_b = %ut_b) -> (!stencil.temp<[0,64]xf64>) {
    %ui = stencil.load(%ui_b) : !stencil.field<[0, 64]xf64> ->  !stencil.temp<[0,64]xf64>
    %unew = stencil.apply(%uarg = %ui : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[0,64]xf64>) {
        %v = stencil.access %uarg[0] : !stencil.temp<[0,64]xf64>
        %vnew = func.call @compute(%v) : (f64) -> f64
        stencil.return %vnews
    }
    %unew_b = stencil.alloc() : ...
    stencil.store(%unew, %unew_b) : ...
    scf.yield(%unew_b)
}
%uout = stencil.load(%uout_b) : ...
stencil.store(%uout, %u) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
```
The last use of `%ut_b` is scf.for, `%u` is untouched including there, so we can just forard it through:
It makes the field bigger so we need to make sure typing is okay... For now, let's assume we can just propagate it to the allocs and all. (I think unrealized casts would do the job here, where one can resolve them by propagating the bigger size)
```mlir
%uout_b = scf.for %time = %0 to %T step %1 iter_args(%ui_b = %u) -> (!stencil.field<[-4,68]xf64>) {
    %ui = stencil.load(%ui_b) : !stencil.field<[0, 64]xf64> ->  !stencil.temp<[0,64]xf64>
    %unew = stencil.apply(%uarg = %ui : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[0,64]xf64>) {
        %v = stencil.access %uarg[0] : !stencil.temp<[0,64]xf64>
        %vnew = func.call @compute(%v) : (f64) -> f64
        stencil.return %vnews
    }
    %unew_b = stencil.alloc() : ...
    stencil.store(%unew, %unew_b) : ...
    scf.yield(%unew_b)
}
%uout = stencil.load(%uout_b) : ...
stencil.store(%uout, %u) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
```

Let's look at the output of scf.for. 

There's a load-store. e load is the last use of its buffer. The store stores to u. So it is safe to instead store to u.

Only difference with the alloc situation is that here, we're talking about the result of a for, not of an alloc.
So instead of just replacing this result with u, we're acting on the yield's operands: we want to yield u with the expected values instead. 
So let's just load/store there! This allows thinking later about where this yielded buffer comes from.

```mlir
%uout_b = scf.for %time = %0 to %T step %1 iter_args(%ui_b = %u) -> (!stencil.field<[-4,68]xf64>) {
    %ui = stencil.load(%ui_b) : !stencil.field<[0, 64]xf64> ->  !stencil.temp<[0,64]xf64>
    %unew = stencil.apply(%uarg = %ui : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[0,64]xf64>) {
        %v = stencil.access %uarg[0] : !stencil.temp<[0,64]xf64>
        %vnew = func.call @compute(%v) : (f64) -> f64
        stencil.return %vnews
    }
    %unew_b = stencil.alloc() : ...
    stencil.store(%unew, %unew_b) : ...
    %uout = stencil.load(%unew_b) : ...
    stencil.store(%uout, %u) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
    scf.yield(%u)
}
```

> Let's just add a corner-case here; we yield %u, which is the initial value of the corresponding iteration argument %ui_b. Let's yield it instead, to avoid aliasing:

```mlir
%uout_b = scf.for %time = %0 to %T step %1 iter_args(%ui_b = %u) -> (!stencil.field<[-4,68]xf64>) {
    %ui = stencil.load(%ui_b) : !stencil.field<[0, 64]xf64> ->  !stencil.temp<[0,64]xf64>
    %unew = stencil.apply(%uarg = %ui : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[0,64]xf64>) {
        %v = stencil.access %uarg[0] : !stencil.temp<[0,64]xf64>
        %vnew = func.call @compute(%v) : (f64) -> f64
        stencil.return %vnews
    }
    %unew_b = stencil.alloc() : ...
    stencil.store(%unew, %unew_b) : ...
    %uout = stencil.load(%unew_b) : ...
    stencil.store(%uout, %ui_b) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
    scf.yield(%ui_b)
}
```

Boom, scf.for bufferized


We can replace loads of just stored values by the value:
```mlir
%uout_b = scf.for %time = %0 to %T step %1 iter_args(%ui_b = %u) -> (!stencil.field<[-4,68]xf64>) {
    %ui = stencil.load(%ui_b) : !stencil.field<[0, 64]xf64> ->  !stencil.temp<[0,64]xf64>
    %unew = stencil.apply(%uarg = %ui : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[0,64]xf64>) {
        %v = stencil.access %uarg[0] : !stencil.temp<[0,64]xf64>
        %vnew = func.call @compute(%v) : (f64) -> f64
        stencil.return %vnews
    }
    %unew_b = stencil.alloc() : ...
    stencil.store(%unew, %unew_b) : ...
    stencil.store(%unew, %ui_b) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
    scf.yield(%ui_b)
}
```
We can simplify allocated buffers that are only written to:

```mlir
%uout_b = scf.for %time = %0 to %T step %1 iter_args(%ui_b = %u) -> (!stencil.field<[-4,68]xf64>) {
    %ui = stencil.load(%ui_b) : !stencil.field<[0, 64]xf64> ->  !stencil.temp<[0,64]xf64>
    %unew = stencil.apply(%uarg = %ui : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[0,64]xf64>) {
        %v = stencil.access %uarg[0] : !stencil.temp<[0,64]xf64>
        %vnew = func.call @compute(%v) : (f64) -> f64
        stencil.return %vnews
    }
    stencil.store(%unew, %ui_b) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
    scf.yield(%ui_b)
}
```

Boom, loop bufferized, it's now reduced to bufferizing the loop's body.

### Bufferswapping

A common pattern for those loops is to swap the buffers between iterations, to just reuse them efficiently amongst iterations.

Let's have a look at how to write it in value-semantics:


```mlir
%ut = stencil.load(%u) : !stencil.field<[-4, 68]xf64> -> !stencil.temp<[0,64]xf64>
%vt = stencil.load(%v) : !stencil.field<[-4, 68]xf64> -> !stencil.temp<[0,64]xf64>
%uout, %vout = scf.for %time = %0 to %T step %1 iter_args(%ui = %ut, %vi = %vt) -> (!stencil.temp<[0,64]xf64>) {
    %vnew = stencil.apply(%uarg = %ui : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[0,64]xf64>) {
        %l = stencil.access %uarg[-1] : !stencil.temp<[0,64]xf64>
        %c = stencil.access %uarg[0] : !stencil.temp<[0,64]xf64>
        %r = stencil.access %uarg[1] : !stencil.temp<[0,64]xf64>
        %vnew = func.call @compute(%l, %c, %r) : (f64, f64, f64) -> f64
        stencil.return %vnews
    }
    scf.yield(%vnew, %ui)
}
stencil.store(%uout, %u) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
stencil.store(%vout, %v) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
```

### More complex case ?

With bufferswapping implemented on the loop:

```mlir
%ut = stencil.load(%u) : !stencil.field<[-4, 68]xf64> -> !stencil.temp<[0,64]xf64>
%vt = stencil.load(%v) : !stencil.field<[-4, 68]xf64> -> !stencil.temp<[0,64]xf64>
%uout, %vout, %ubout, %vbout = scf.for %time = %0 to %T step %1 iter_args(%ui = %ut, %vi = %vt, %ub = %u, %vb = %v) -> (!stencil.temp<[0,64]xf64>) {
    %vnew = stencil.apply(%uarg = %ui : !stencil.temp<[0,64]xf64>) -> (!stencil.temp<[0,64]xf64>) {
        %l = stencil.access %uarg[-1] : !stencil.temp<[0,64]xf64>
        %c = stencil.access %uarg[0] : !stencil.temp<[0,64]xf64>
        %r = stencil.access %uarg[1] : !stencil.temp<[0,64]xf64>
        %vnew = func.call @compute(%l, %c, %r) : (f64, f64, f64) -> f64
        stencil.return %vnews
    }
    scf.yield(%vnew, %ui, %v, %u)
}
stencil.store(%uout, %ubout) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
stencil.store(%vout, %vbout) : !stencil.temp<[0,64]xf64> to !stencil.field<[-4,68]xf64>
```
