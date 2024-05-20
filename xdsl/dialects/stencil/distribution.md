# Stencil distribution and value-semantics

Let's consider local iterated stencils in value-semantics sorted here.

The distribution question remains.

Let's go over how it's done in mixed semantics currently:

Here's the example:
```mlir
// Iterate %N time, take initial buffers
    scf.for %time = %0 to %T step %1 iter_args(%u = %u_init, %v = %v_init) -> (!stencil.field<[-4,68]x[-4,68]xf64>, !stencil.field<[-4,68]x[-4,68]xf64>) {

        // Load input
        %ut = stencil.load %u : !stencil.field<[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?xf64>

        // Value-semantics compute
        %vt = stencil.apply(%uarg = %ut : !stencil.temp<?x?xf64>) -> (!stencil.temp<?x?xf64>) {
            %left = stencil.access %uarg[-2, 0] : !stencil.temp<?x?xf64>
            %center = stencil.access %uarg[0, 0] : !stencil.temp<?x?xf64>
            %right = stencil.access %uarg[2, 0] : !stencil.temp<?x?xf64>
            %value = func.call @compute(%left, %center, %right) : (f64, f64, f64) -> f64
            stencil.return %value : f64
        }

        // Store outputs
        stencil.store %vt to %v ([0, 0] : [64, 64]) : !stencil.temp<?x?xf64> to !stencil.field<[-4,68]x[-4,68]xf64>

        // Swap buffers
        scf.yield %v, %u : !stencil.field<[-4,68]x[-4,68]xf64>, !stencil.field<[-4,68]x[-4,68]xf64>

    }
```
Here's the output of `distribute-stencil{slices=2,1 strategy=2d-grid restrict-domain=true}`:

```mlir
%2, %3 = scf.for %time = %0 to %T step %1 iter_args(%u = %u_init, %v = %v_init) -> (!stencil.field<[-4,68]x[-4,68]xf64>, !stencil.field<[-4,68]x[-4,68]xf64>) {

    %ut = stencil.load %u : !stencil.field<[-4,68]x[-4,68]xf64> -> !stencil.temp<[-1,33]x[0,64]xf64>

    "dmp.swap"(%ut) {"topo" = #dmp.topo<2x1>, "swaps" = [#dmp.exchange<at [32, 0] size [1, 64] source offset [-1, 0] to [1, 0]>, #dmp.exchange<at [-1, 0] size [1, 64] source offset [1, 0] to [-1, 0]>]} : (!stencil.temp<[-1,33]x[0,64]xf64>) -> ()

    %vt = stencil.apply(%uarg = %ut : !stencil.temp<[-1,33]x[0,64]xf64>) -> (!stencil.temp<[0,32]x[0,64]xf64>) {
        %left = stencil.access %uarg[-1, 0] : !stencil.temp<[-1,33]x[0,64]xf64>
        %center = stencil.access %uarg[0, 0] : !stencil.temp<[-1,33]x[0,64]xf64>
        %right = stencil.access %uarg[1, 0] : !stencil.temp<[-1,33]x[0,64]xf64>
        %value = func.call @compute(%left, %center, %right) : (f64, f64, f64) -> f64
        stencil.return %value : f64
    }

    stencil.store %vt to %v ([0, 0] : [32, 64]) : !stencil.temp<[0,32]x[0,64]xf64> to !stencil.field<[-4,68]x[-4,68]xf64>

    scf.yield %v, %u : !stencil.field<[-4,68]x[-4,68]xf64>, !stencil.field<[-4,68]x[-4,68]xf64>
    }
```

A few comments:

- This work on the passed buffer sizes; Automatically splitting those yields different integration problems, so we'll keep working with that for now.
- This ties to the loads that we want out of the way.
- This only work on simple cases. Let's see what can break it and what it implies next.
- It acts on value-semantic temps but returns nothing! It's a hack, it should return a value representing the swapped values.


Let's assume all computations are done on a same coordinate domain; otherwise, we have deeper problems, can explain later!

In that assumption, we know we want to split all computations similarly, according to a known grid topology.
So, we know we simply have to exchange halos when input space-dependencies cross the split dimensions in that topology! (i.e, we have a dependency accros dimension n and n is split in the topology.)

I want to first split things a bit and just remove the tie to stores and loads. This already will ;ake the distribution applicable to way wider class of stencil programs.

Let's start looking at our iterated example:

```mlir
// Load (core) inputs
%u_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%v_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>

// Load (ghost) inputs
%ul_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%uu_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%vl_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%vu_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>

// Iterate %N time, take initial buffers
scf.for %time = %0 to %T step %1 iter_args(%ul = %ul_init, %u = %u_init, %uu = %uu_init, %vl = %vl_init, %v = %v_init, %vu = %vu_init) -> (!stencil.temp<?xf64>, !stencil.field<?xf64>, !stencil.temp<?xf64>, !stencil.field<?xf64>, !stencil.temp<?xf64>, !stencil.field<?xf64>) {

    // Value-semantics ghost cells reading
    %u1 = stencil.combine 0 at 64 lower = (%u : !stencil.temp<?xf64>), upper = (%uu : !stencil.temp<?xf64>) : !stencil.temp<?xf64> 
    %uh = stencil.combine 0 at 0 lower = (%ul : !stencil.temp<?xf64>), upper = (%u1 : !stencil.temp<?xf64>) : !stencil.temp<?xf64> 

    // Value-semantics compute
    %vt = stencil.apply(%uarg = %uh : !stencil.field<?xf64>) -> (!stencil.temp<?xf64>) {
        %left = stencil.access %uarg[-1] : !stencil.temp<?xf64>
        %center = stencil.access %uarg[0] : !stencil.temp<?xf64>
        %right = stencil.access %uarg[1] : !stencil.temp<?xf64>
        %value = func.call @compute(%left, %center, %right) : (f64, f64, f64) -> f64
        stencil.return %value : f64
    }
    // Value-semantics swap
    scf.yield(%vl, %v, %vr, %ul, %u, %uu)
}
// Store outputs
stencil.store %uout to %ub ([0] : [64]) : !stencil.temp<?xf64> to  !stencil.field<[-4,68]xf64>
stencil.store %vout to %vb ([0] : [64]) : !stencil.temp<?xf64> to  !stencil.field<[-4,68]xf64>
```

Let's *independently* split the stores, AKA, split the effective computation needed:

```mlir
// Load (core) inputs
%u_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%v_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>

// Load (ghost) inputs
%ul_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%uu_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%vl_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%vu_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>

// Iterate %N time, take initial buffers
scf.for %time = %0 to %T step %1 iter_args(%ul = %ul_init, %u = %u_init, %uu = %uu_init, %vl = %vl_init, %v = %v_init, %vu = %vu_init) -> (!stencil.temp<?xf64>, !stencil.field<?xf64>, !stencil.temp<?xf64>, !stencil.field<?xf64>, !stencil.temp<?xf64>, !stencil.field<?xf64>) {

    // Value-semantics ghost cells reading
    %u1 = stencil.combine 0 at 64 lower = (%u : !stencil.temp<?xf64>), upper = (%uu : !stencil.temp<?xf64>) : !stencil.temp<?xf64> 
    %uh = stencil.combine 0 at 0 lower = (%ul : !stencil.temp<?xf64>), upper = (%u1 : !stencil.temp<?xf64>) : !stencil.temp<?xf64> 

    // Value-semantics compute
    %vt = stencil.apply(%uarg = %uh : !stencil.field<?xf64>) -> (!stencil.temp<?xf64>) {
        %left = stencil.access %uarg[-1] : !stencil.temp<?xf64>
        %center = stencil.access %uarg[0] : !stencil.temp<?xf64>
        %right = stencil.access %uarg[1] : !stencil.temp<?xf64>
        %value = func.call @compute(%left, %center, %right) : (f64, f64, f64) -> f64
        stencil.return %value : f64
    }
    // Value-semantics swap
    scf.yield(%vl, %v, %vr, %ul, %u, %uu)
}
// Store outputs
stencil.store %uout to %ub ([0] : [32]) : !stencil.temp<?xf64> to  !stencil.field<[-4,68]xf64>
stencil.store %vout to %vb ([0] : [32]) : !stencil.temp<?xf64> to  !stencil.field<[-4,68]xf64>
```

Oh nope, the `stencil.combine`-based everygrowing guard is not split. Worst than this, we just don't know how to split it locally, as its just stitching two temps on one fixed coordinate.
A solution would be to use the mentionned `stencil.halo`, having a clearer definition of what's "core" and what's "halo" instead:

```mlir
// Load (core) inputs
%u_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%v_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>

// Load (ghost) inputs
%ul_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%uu_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%vl_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%vu_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>

// Iterate %N time, take initial buffers
scf.for %time = %0 to %T step %1 iter_args(%ul = %ul_init, %u = %u_init, %uu = %uu_init, %vl = %vl_init, %v = %v_init, %vu = %vu_init) -> (!stencil.temp<?xf64>, !stencil.field<?xf64>, !stencil.temp<?xf64>, !stencil.field<?xf64>, !stencil.temp<?xf64>, !stencil.field<?xf64>) {

    // Value-semantics ghost cells reading

    %uh = stencil.halo 0 [%ul|0|%u|64|%ul] : !stencil.temp<?xf64> -> !stencil.temp<?xf64>

    // Value-semantics compute
    %vt = stencil.apply(%uarg = %uh : !stencil.field<?xf64>) -> (!stencil.temp<?xf64>) {
        %left = stencil.access %uarg[-1] : !stencil.temp<?xf64>
        %center = stencil.access %uarg[0] : !stencil.temp<?xf64>
        %right = stencil.access %uarg[1] : !stencil.temp<?xf64>
        %value = func.call @compute(%left, %center, %right) : (f64, f64, f64) -> f64
        stencil.return %value : f64
    }
    // Value-semantics swap
    scf.yield(%vl, %v, %vr, %ul, %u, %uu)
}
// Store outputs
stencil.store %uout to %ub ([0] : [64]) : !stencil.temp<?xf64> to  !stencil.field<[-4,68]xf64>
stencil.store %vout to %vb ([0] : [64]) : !stencil.temp<?xf64> to  !stencil.field<[-4,68]xf64>
```

The assumption now is that all computations are done on a same grid (formalization later), and all halos share those same coordiantes. Thus, we can just split the halos the same way we split the stores (?):

```mlir
// Load (core) inputs
%u_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%v_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>

// Load (ghost) inputs
%ul_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%uu_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%vl_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%vu_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>

// Iterate %N time, take initial buffers
scf.for %time = %0 to %T step %1 iter_args(%ul = %ul_init, %u = %u_init, %uu = %uu_init, %vl = %vl_init, %v = %v_init, %vu = %vu_init) -> (!stencil.temp<?xf64>, !stencil.field<?xf64>, !stencil.temp<?xf64>, !stencil.field<?xf64>, !stencil.temp<?xf64>, !stencil.field<?xf64>) {

    // Value-semantics ghost cells reading

    %uh = stencil.halo 0 [%ul|0|%u|32|%ul] : !stencil.temp<?xf64> -> !stencil.temp<?xf64>

    // Value-semantics compute
    %vt = stencil.apply(%uarg = %uh : !stencil.field<?xf64>) -> (!stencil.temp<?xf64>) {
        %left = stencil.access %uarg[-1] : !stencil.temp<?xf64>
        %center = stencil.access %uarg[0] : !stencil.temp<?xf64>
        %right = stencil.access %uarg[1] : !stencil.temp<?xf64>
        %value = func.call @compute(%left, %center, %right) : (f64, f64, f64) -> f64
        stencil.return %value : f64
    }
    // Value-semantics swap
    scf.yield(%vl, %v, %vr, %ul, %u, %uu)
}
// Store outputs
stencil.store %uout to %ub ([0] : [32]) : !stencil.temp<?xf64> to  !stencil.field<[-4,68]xf64>
stencil.store %vout to %vb ([0] : [32]) : !stencil.temp<?xf64> to  !stencil.field<[-4,68]xf64>
```
Let's trigger typical shape inference again: 


```mlir
// Load (core) inputs
%u_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%v_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>

// Load (ghost) inputs
%ul_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%uu_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%vl_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%vu_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>

// Iterate %N time, take initial buffers
scf.for %time = %0 to %T step %1 iter_args(%ul = %ul_init, %u = %u_init, %uu = %uu_init, %vl = %vl_init, %v = %v_init, %vu = %vu_init) -> (!stencil.temp<[-1,0]xf64>, !stencil.field<[0,32]xf64>, !stencil.temp<[32,33]xf64>, !stencil.field<[-1,0]xf64>, !stencil.temp<[0,32]xf64>, !stencil.field<[32,33]xf64>) {

    // Value-semantics ghost cells reading

    %uh = stencil.halo 0 [%ul|0|%u|32|%ul] : !stencil.temp<[0,32]xf64> -> !stencil.temp<[-1,33]xf64>

    // Value-semantics compute
    %vt = stencil.apply(%uarg = %uh : !stencil.temp<[-1,33]xf64>) -> (!stencil.tem[0,32]<?xf64>) {
        %left = stencil.access %uarg[-1] : !stencil.temp<[-1,33]xf64>
        %center = stencil.access %uarg[0] : !stencil.temp<[-1,33]xf64>
        %right = stencil.access %uarg[1] : !stencil.temp<[-1,33]xf64>
        %value = func.call @compute(%left, %center, %right) : (f64, f64, f64) -> f64
        stencil.return %value : f64
    }
    // Value-semantics swap
    scf.yield(%vl, %v, %vr, %ul, %u, %uu)
}
// Store outputs
stencil.store %uout to %ub ([0] : [32]) : !stencil.temp<[0,32]xf64> to  !stencil.field<[-4,68]xf64>
stencil.store %vout to %vb ([0] : [32]) : !stencil.temp<[0,32]xf64> to  !stencil.field<[-4,68]xf64>
```

Worked out nicely!
Now, we can independently just insert swaps on applys' inputs, according to a same grid, and seeing the expected coordinates of the exchanges and access patterns: 
```mlir
// Load (core) inputs
%u_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%v_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>

// Load (ghost) inputs
%ul_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%uu_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%vl_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%vu_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>

// Iterate %N time, take initial buffers
scf.for %time = %0 to %T step %1 iter_args(%ul = %ul_init, %u = %u_init, %uu = %uu_init, %vl = %vl_init, %v = %v_init, %vu = %vu_init) -> (!stencil.temp<[-1,0]xf64>, !stencil.field<[0,32]xf64>, !stencil.temp<[32,33]xf64>, !stencil.field<[-1,0]xf64>, !stencil.temp<[0,32]xf64>, !stencil.field<[32,33]xf64>) {

    // Value-semantics ghost cells reading
    %uh = stencil.halo 0 [%ul|0|%u|32|%ul] : !stencil.temp<[0,32]xf64> -> !stencil.temp<[-1,33]xf64>

    // Value-semantics halo swap
    %uh_1 = "dmp.swap"(%uh) {"topo" = #dmp.topo<2x1>, "swaps" = [#dmp.exchange<at [32, 0] size [1, 64] source offset [-1, 0] to [1, 0]>, #dmp.exchange<at [-1, 0] size [1, 64] source offset [1, 0] to [-1, 0]>]} : (!stencil.temp<[-1,33]x[0,64]xf64>) -> ()

    // Value-semantics compute
    %vt = stencil.apply(%uarg = %uh_1 : !stencil.temp<[-1,33]xf64>) -> (!stencil.tem[0,32]<?xf64>) {
        %left = stencil.access %uarg[-1] : !stencil.temp<[-1,33]xf64>
        %center = stencil.access %uarg[0] : !stencil.temp<[-1,33]xf64>
        %right = stencil.access %uarg[1] : !stencil.temp<[-1,33]xf64>
        %value = func.call @compute(%left, %center, %right) : (f64, f64, f64) -> f64
        stencil.return %value : f64
    }
    // Value-semantics swap
    scf.yield(%vl, %v, %vr, %ul, %u, %uu)
}
// Store outputs
stencil.store %uout to %ub ([0] : [32]) : !stencil.temp<[0,32]xf64> to  !stencil.field<[-4,68]xf64>
stencil.store %vout to %vb ([0] : [32]) : !stencil.temp<[0,32]xf64> to  !stencil.field<[-4,68]xf64>
```

And we know the rest of the lowering.

Gained:
- Distribute chains of stencils, not only at load and store boundaries.
- Proper value-semantics: we can now easily think about exploding the computation on the diffent bits, e.g., work last on the swapped dependencies for communication computation overlap or what not.
