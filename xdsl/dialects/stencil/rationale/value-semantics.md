# Value-semantic stencil programs

## The problem
Currently, we iterate stencils in mixed semantics as follows:

```mlir
//                             Iterate %N time          take initial buffers %u_init and %v_init
//                             vvvvvvvvvvvvvvvv          vvvvvvvvvvvvvvvvvvvvvvvvvvvv
%uout, %vout = scf.for %time = %0 to %T step %1 iter_args(%u = %u_init, %v = %v_init) -> (!stencil.field<[-4,68]xf64>, !stencil.field<[-4,68]xf64>) {

    // Load input
    %ut = stencil.load %u : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>

    // Value-semantics compute
    %vt = stencil.apply(%uarg = %ut : !stencil.temp<?xf64>) -> (!stencil.temp<?xf64>) {
        %left = stencil.access %uarg[-1] : !stencil.temp<?xf64>
        %center = stencil.access %uarg[0] : !stencil.temp<?xf64>
        %right = stencil.access %uarg[1] : !stencil.temp<?xf64>
        %value = func.call @compute(%left, %center, %right) : (f64, f64, f64) -> f64
        stencil.return %value : f64
    }

    // Store outputs
    stencil.store %vt to %v ([0] : [64]) : !stencil.temp<?xf64> to  !stencil.field<[-4,68]xf64>

    // Swap buffers
    scf.yield(%v, %u)

}
```

This is self-contained and work well with shape inference illustrated below, which does not have to go beyond the iteration (read bottom-up to follow the shape inference's order):


```mlir
// Iterate %N time, take initial buffers
scf.for %time = %0 to %T step %1 iter_args(%u = %u_init, %v = %v_init) -> (!stencil.field<[-4,68]xf64>, !stencil.field<[-4,68]xf64>) {

    // 3. Okay, buffer %u big enough, job done.
    %ut = stencil.load %u : !stencil.field<[-4,68]xf64> -> !stencil.temp<[-1,65]xf64>

    // 2. Pull [-1,65] from %ut, as that's the accessed portion to compute the requested %vt
    %vt = stencil.apply(%uarg = %ut : !stencil.temp<[-1,65]xf64>) -> stencil.temps<[0,64]xf64> {
        %left = stencil.access %uarg[-1] : !stencil.temp<[-1,65]xf64>
        %center = stencil.access %uarg[0] : !stencil.temp<[-1,65]xf64>
        %right = stencil.access %uarg[1] : !stencil.temp<[-1,65]xf64>
        %value = func.call @compute(%left, %center, %right) : (f64, f64, f64) -> f64
        stencil.return %value : f64
    }

    // 1. Pull [0,64] from %vt
    stencil.store %vt to %v ([0] : [64]) : !stencil.temp<[0,64]xf64> to  !stencil.field<[-4,68]xf64>

    // Swap buffers
    scf.yield(%v, %u)

}
```

Now, we want to move to value-semantics iteration. It would yield nice optimizations basically for free!

Let's naively try:


```mlir
// Load input
%u_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%v_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>

//                             Iterate %N time          take initial *values* %u_init and %v_init
//                             vvvvvvvvvvvvvvvv          vvvvvvvvvvvvvvvvvvvvvvvvvvvv
%uout, %vout = scf.for %time = %0 to %T step %1 iter_args(%u = %u_init, %v = %v_init) -> (!stencil.temp<?xf64>, !stencil.field<?xf64>) {

    // Value-semantics compute
    %vt = stencil.apply(%uarg = %u : !stencil.field<?xf64>) -> (!stencil.temp<?xf64>) {
        %left = stencil.access %uarg[-1] : !stencil.temp<?xf64>
        %center = stencil.access %uarg[0] : !stencil.temp<?xf64>
        %right = stencil.access %uarg[1] : !stencil.temp<?xf64>
        %value = func.call @compute(%left, %center, %right) : (f64, f64, f64) -> f64
        stencil.return %value : f64
    }
    // Value-semantics swap
    scf.yield(%v, %u)
}
// Store outputs
stencil.store %uout to %ub ([0] : [64]) : !stencil.temp<?xf64> to  !stencil.field<[-4,68]xf64>
stencil.store %vout to %vb ([0] : [64]) : !stencil.temp<?xf64> to  !stencil.field<[-4,68]xf64>
```

This would trigger an endless shape inference:

```mlir
// Load inputs
%u_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%v_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>

// 2. Pull [0, 64] on scf.yield's operands, %v_new and %u.
// 4. This requires pulling from %v_new in the yield again, but [-1, 65]. Endless loop starting!
%uout, %vout = scf.for %time = %0 to %T step %1 iter_args(%u = %u_init, %v = %v_init) -> (!stencil.temp<[-1,65]xf64>, !stencil.field<[0,64]xf64>) {

    // 3. Pull [-1,65] from %u to compute %v_new
    %v_new = stencil.apply(%uarg = %u : !stencil.field<[-1,65]xf64>) -> (!stencil.temp<[0,64]xf64>) {
        %left = stencil.access %uarg[-1] : !stencil.temp<?xf64>
        %center = stencil.access %uarg[0] : !stencil.temp<?xf64>
        %right = stencil.access %uarg[1] : !stencil.temp<?xf64>
        %value = func.call @compute(%left, %center, %right) : (f64, f64, f64) -> f64
        stencil.return %value : f64
    }

    // Value-semantics swap
    scf.yield(%v_new, %u)
}

// 1. Pull [0, 64] from %uout and %vout
stencil.store %uout to %ub ([0] : [64]) : !stencil.temp<?xf64> to  !stencil.field<[-4,68]xf64>
stencil.store %vout to %vb ([0] : [64]) : !stencil.temp<?xf64> to  !stencil.field<[-4,68]xf64>
```

So what's missing?

One could see that the first stencil IR encodes a common pattern: We update values in `[0,64]` but read the input beyond that, and those values beyond are never updated (by the main stencil computation, at least). Depending on the roles of these values, they are most commonly referred to as "halo" (most common for read-only) or "ghost cells"(most common when updated differently, encoding boundary conditions).
We often refer to the region without halo (`[0, 64]`, in our example) as the *core*

In mixed-semantics, `stencil.store` played the role of encoding the core bounds. `stencil.load`, through its reference-semantics, played the role of forwarding whatever values were already in the buffer beyond those bounds. By pulling them out of the loop, we lost that information!

Thinking 1D, the problem we have is that we want to compute some values, and later access those newly computed values along with older values on the "boundary":

```
                                                                  
┌────────────────┬────────────────────────────┬─────────────────┐ 
│left "boundary" │       computed/core        │ right "boundary"│
└────────────────┴────────────────────────────┴─────────────────┘ 
▲                ▲                            ▲                 ▲ 
│                └──────────────┬─────────────┘                 │ 
│                               │                               │ 
│          We want to define or constrain those boundaries      │ 
│                                                               │ 
│              Let's name them [core_low, core_up]              │ 
│                                                               │ 
└───────────────────────────────┬───────────────────────────────┘ 
                                │                                 
                    We know those boundaries.                     
                 Those are the one requested by the               
                    operation using the values.                                
                                                                  
               Let's name them [user_low, user_up]                
```

                                                                                      
          
                                                  
So we have        
```
│core_low = user_low + halo_low│
│core_up  = user_up  + halo_up │
```
And we only know `user_low` and `user_up` locally.
So we have to define either `halo_low`&`halo_up`, or `core_low`&`core_up`.                 
                                                                                      
The former is defining a halo width, the latter is defining a core size.
They are equivalent, so it boils down to user preference, or just allowing both.\
We already have a mechanism to define a known core_low/core_up in stencil.combine,
so the following will use that first. It also just sounds more consistent with the current role of `stencil.store`.

## A first minimal solution : `stencil.combine`

`stencil.combine` takes two stencil temps and combine them in one temp, around a cutoff coordinate on a given dimension.

`%res1, %res2 = stencil.combine 1 at 11 lower = (%0 : !stencil.temp<?x?x?xf64>) upper = (%1 : !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>`

Can be illustrated by:

```
        dim   1       offset       
             ┌──►      (=11)       
           0 │          │          
             ▼ ┌────────┼─────────┐
               │        │         │
               │        │         │
          %res1│  lower │ upper   │
               │    %0  │   %1    │
               │        │         │
               │        │         │
               └────────┼─────────┘
                        │          
```

Let's stick to 1D for the following.
Here, we want to express a fixed-sized (`[<lb>] : [<ub>]`) core values (`%core`) extended by ghost cells (`%u` and `%v`), i.e.:

```
                                       │      ┌──┐     ┌─────────────────────┐     ┌──┐                   
                                       │      │%l│     │        %core        │     │%u│                   
                                       │      └──┘     └─────────────────────┘     └──┘                   
                                       │                                                                  
                                       │                                                                  
                                       │                                                                  
                                       │                                                                  
      known core bounds                │     %coreu = stencil.combine 0 at <ub> lower = %core, upper = %u 
    ◄───────────────────►              │                                                                  
                                       │                                     ub                           
  lb                     ub            │                                     │                            
   │                     │             │      ┌──┐     ┌─────────────────────┼──┐                         
┌──┼─────────────────────┼──┐          │      │%l│     │        %core        │%u│                         
│%l│        %core        │%u│   ─────► │      └──┘     └─────────────────────┼──┘                         
└──┼─────────────────────┼──┘          │                                     │                            
   │                     │             │                                                                  
                                       │                                                                  
 ◄─────────────────────────►           │                                                                  
     What the computation              │     %coreh = stencil.combine 0 at <lb> lower = %l, upper = %coreu
     wants to access                   │                                                                  
                                       │              lb                                                  
                                       │               │                                                  
                                       │            ┌──┼─────────────────────┬──┐                         
                                       │            │%l│        %core        │%u│                         
                                       │            └──┼─────────────────────┴──┘                         
                                       │               │                                                  
```

Looks like we can at least express it. Let's try in a loop?

Because we want to avoid the evergrowing of the "core" part of the stencil, let's load from the input fields multiple times, once for the core, and once for each ghost side. Otherwise, when growing the combined size, we would grow the core value's size too.
Because we want the ghost cells to be swapped along with the core values, let's add those ghost loads to the iteration arguments to express their swap.

```mlir
// Load (core) inputs
%u_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%v_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>

// Load (ghost) inputs
%ul_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%uu_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%vl_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>
%vu_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<?xf64>

// Iterate %N time, take initial values for cores and halos
%ulout, %uout, %uuout, %vlout, %vout, %vuout = scf.for %time = %0 to %T step %1 iter_args(%ul = %ul_init, %u = %u_init, %uu = %uu_init, %vl = %vl_init, %v = %v_init, %vu = %vu_init) -> (!stencil.temp<?xf64>, !stencil.temp<?xf64>, !stencil.temp<?xf64>, !stencil.temp<?xf64>, !stencil.temp<?xf64>, !stencil.temp<?xf64>) {

    // Stitch together the input core and halos
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

Let's try a shape inference pass on this thing:

```mlir
// 8. Just request the right loads. Everything is in bound! Job done.
%u_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<[0,64]xf64>
%v_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<[0,64]xf64>

%ul_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<[-1,0]xf64>
%uu_init = stencil.load %ub : !stencil.field<[-4,68]xf64> -> !stencil.temp<[64,65]xf64>
%vl_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<[-1,0]xf64>
%vu_init = stencil.load %vb : !stencil.field<[-4,68]xf64> -> !stencil.temp<[64,65]xf64>

// 2. Pull [0,64] from scf.yield's %vt and %u

// 6. %u was pulled [0,64] from both yield and apply, %ul and %uu were pulled sound bounds too, [-1, 0] and [64, 65] respectively.
// -> Pull [-1,0], [64,65] from yield's %vl, %vu; [0,64] from %vt already pulled :)

// 7. Those are just taken from the arguments of for though so directly reflect them, and now everything is matching.
// We can thus pull from scf.for's operands.
%ulout, %uout, %uuout, %vlout, %vout, %vuout = scf.for %time = %0 to %T step %1 iter_args(%ul = %ul_init, %u = %u_init, %uu = %uu_init, %vl = %vl_init, %v = %v_init, %vu = %vu_init) -> (!stencil.temp<[-1,0]xf64>, !stencil.temp<[0,64]xf64>, !stencil.temp<[64,65]xf64>, !stencil.temp<[-1,0]xf64>, !stencil.temp<[0,64]xf64>, !stencil.temp<[64,65]xf64>) {

    // 5. There's a cutoff at 64, so pull [0,64] from %u and [64,65] from %uu
    %u1 = stencil.combine 0 at 64 lower = (%u : !stencil.temp<[0,64]xf64>), upper = (%uu : !stencil.temp<[64,65]xf64>) : !stencil.temp<[0,65]xf64>

    //4. There's a cutoff at 0, so pull [-1,0] from %ul and [0,65] from %u1
    %uh = stencil.combine 0 at 0 lower = (%ul : !stencil.temp<[-1,0]xf64>), upper = (%u1 : !stencil.temp<[0,65]xf64>) : !stencil.temp<[-1,65]xf64>

    // 3. Pull [-1,65] from %uh to provide %vt's [0,64]
    %vt = stencil.apply(%uarg = %uh : !stencil.temp<[-1,65]xf64>) -> (!stencil.temp<[0,64]xf64>) {
        %left = stencil.access %uarg[-1] : !stencil.temp<[-1,65]xf64>
        %center = stencil.access %uarg[0] : !stencil.temp<[-1,65]xf64>
        %right = stencil.access %uarg[1] : !stencil.temp<[-1,65]xf64>
        %value = func.call @compute(%left, %center, %right) : (f64, f64, f64) -> f64
        stencil.return %value : f64
    }

    scf.yield(%vl, %vt, %vu, %ul, %u, %uu)
}
// 1. Pull [0,64] from %uout and %vout
stencil.store %uout to %ub ([0] : [64]) : !stencil.temp<[0,64]xf64> to  !stencil.field<[-4,68]xf64>
stencil.store %vout to %vb ([0] : [64]) : !stencil.temp<[0,64]xf64> to  !stencil.field<[-4,68]xf64>
```

So it sounds like shape inference can be easily adapted!

### Open ends

- [value-lowering.md](): This thing doesn't even verify in the original dialect. It violates restrictions that make the lowering to buffer semantic trivial. The lowering has to be thought again if we want this thing to lower ever.

- [stencil-combine.md](): This `stencil.combine` looks too low-level for this. We want something more high-level to think about different ways to access ghost cells/halos.

- [distribution.md]: Right now, the distribution depends on stores and loads, i.e, non-value-semantics operations. We need to rethink this process in those terms for this all to make sense.

