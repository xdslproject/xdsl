# Higher-level stencil combination/halo

Currently, the stencil dialect only exposes `stencil.combine` to combine different stencil values:
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

Even in 1D, this requires 2 combines for a typical halo access expression. It gets tedious to express various communication patterns this way.
Using https://arxiv.org/pdf/2312.13094, Fig. 5 as a reference:

Basic:
```
┌──────────────────────────┐
│        step B            │
├───┬──────────────────┬───┤
│   │                  │   │
│   │                  │   │
│ s │                  │ s │
│ t │                  │ t │
│ e │       core       │ e │
│ p │                  │ p │
│   │                  │   │
│ A │                  │ A │
│   │                  │   │
├───┴──────────────────┴───┤
│        step B            │
└──────────────────────────┘
```

So "step A" is a pair of combines as illustrated above, and "step B" a second pair.

Diagonal:
```
┌───┬──────────────────┬───┐
│ D │    step B        │ D │
├───┼──────────────────┼───┤
│   │                  │   │
│   │                  │   │
│ s │                  │ s │
│ t │                  │ t │
│ e │     core         │ e │
│ p │                  │ p │
│   │                  │   │
│ A │                  │ A │
│   │                  │   │
├───┼──────────────────┼───┤
│ D │    step B        │ D │
└───┴──────────────────┴───┘
```
We could either combine "D"s to "step A"s or "step B"s. sturctural complexity-wise, it's equivalent, so let's go with combine to B:

coreA = A + core + A is one such pair of combines.

BD = D + B + D is another such pair of combines.

Finally, BD + coreA + BD is another pair.


As far as I understand, the "Full" mode, at its core, is all about splitting the computation domain too, beyond just the data:
```
┌───┬──────────────────┬───┐
│ D │    step B        │ D │
├───┼──┬────────────┬──┼───┤
│   │R1│    R2      │R3│   │
│   ├──┼────────────┼──┤   │
│ s │  │            │  │ s │
│ t │  │            │  │ t │
│ e │R4│  core      │R5│ e │
│ p │  │            │  │ p │
│   │  │            │  │   │
│ A ├──┼────────────┼──┤ A │
│   │R6│    R7      │R8│   │
├───┼──┴────────────┴──┼───┤
│ D │    step B        │ D │
└───┴──────────────────┴───┘
```

The only remaining complexity being to first compute "core", then wait for the halos before computing the Rs.

We can see the composition pattern of the computed aread being exactly the same one. So, we can just express R1,...R8 and core as different `stencil.apply` and combine the output into the output buffer!

So, let's think about an operation epressing such combinations at a higher level, shall we?


Let's look again at the Diagonal case, which generalizes Basic nicely. Let's just rename pieces
```
┌───┬──────────────────┬───┐
│ 0 │        1         │ 2 │
├───┼──────────────────┼───┤
│   │                  │   │
│   │                  │   │
│   │                  │   │
│   │                  │   │
│ 3 │     core         │ 4 │
│   │                  │   │
│   │                  │   │
│   │                  │   │
│   │                  │   │
├───┼──────────────────┼───┤
│ 5 │       6          │ 7 │
└───┴──────────────────┴───┘
```

In 1D, we have the core and the 2 outter "vertices" : 3 operands.\
In 2D, we have the core, the 4 "sides" and the 4 "vertices" : 9 operands.\
In 3D, we have the core, the 6 "faces", the 12 "edges" and the 8 "vertices" : 27 operands.\
...\
In nD, we have $ 3^n $ operands.
🤕

A bit much IMO.

To backtrack a bit on the reasoning:

While reconstructing the figure from a graph of combines sounds a bit much; when would we actually want to do that? I think the only place it practically makes sense in currently, is when distributing a stencil computation. In this case, we already are analyzing a global stencil computation to construct such figures, to express exchanges. Maybe we just want to change that? From that point generating a lot of combines is trivial recursion.

Otherwise, something that would simplify this very muchg while being less a strech IMO, is a 3-fold equivalent of `stencil.combine`. let's say `stencil.extend`, and make it work exactly as `stencil.combine for now, i.e., it works like a pair of combines:

```
                                       │      ┌──┐     ┌─────────────────────┐     ┌──┐        
                                       │      │%l│     │        %core        │     │%u│        
                                       │      └──┘     └─────────────────────┘     └──┘        
      known core bounds                │                                                       
   ◄─────────────────────►             │                                                       
                                       │    %coreh = stencil.extend dim 0 %l <lb> %core <ub> %u
  lb                     ub            │                                                       
   │                     │             │                                                       
┌──┼─────────────────────┼──┐          │                                                       
│%l│        %core        │%u│   ─────► │              lb                     ub                
└──┼─────────────────────┼──┘          │               │                     │                 
   │                     │             │            ┌──┼─────────────────────┼──┐              
                                       │            │%l│        %core        │%u│              
◄───────────────────────────►          │            └──┼─────────────────────┼──┘              
     What the computation              │               │                     │                 
     wants to access                   │
```

Different assemblies of this could express all mentionned patterns.