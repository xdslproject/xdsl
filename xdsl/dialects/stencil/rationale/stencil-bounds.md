# Stencil bounds and cast

The original stencil dialect encoded bounds as attributes on operations defining bounds, and only sizes on its types.

Here's an example IR before shape inference:

```mlir
func @simple(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>

  %2 = stencil.load %0 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>

  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %4 = stencil.access %arg2 [-1, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = stencil.access %arg2 [1, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %6 = addf %4, %5 : f64
    stencil.return %6 : !stencil.result<f64>
  }

  stencil.store %3 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>

  return
}
```

First of all, this explains the necessity of `stencil.cast`, which is now accessory in xDSL's port - and might just be removed soon.

It actually encodes the bounds in stencil-coordinates on the buffers, as the types would only give sizes, and no origin.


> NB: A common misconception is to assume this allows to use the stencil dialect with dynamic sizes somehow. `stencil.cast` was lowered to `memref.cast`. Citing MLIR documentation:\
\
 "If the cast converts any dimensions from an unknown to a known size, then it acts as an assertion that fails at runtime if the dynamic dimensions disagree with resultant destination size."\
 \
It got worse since then; lowering the above example in modern MLIR typically would fold away various memref aliasing ops though, removing any such runtime checks and generating code acting on dynamic sized memrefs.\
It breaks the cheap access arithmetic promise, *and* generate code accepting any size but *crashing* on too-small buffers and not computing on the full buffers if passed bigger ones.

Here's is OEC's encoding of the IR after shape inference: see the values now having fixed sizes, and the added bounds as an attribute on `stencil.load` and `stencil.apply`:

```mlir
func @simple(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.load %0([-1, -2, 0] : [65, 66, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<66x68x60xf64>
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<66x68x60xf64>) -> !stencil.temp<64x64x60xf64> {
    %4 = stencil.access %arg2 [-1, 2, 0] : (!stencil.temp<66x68x60xf64>) -> f64
    %5 = stencil.access %arg2 [1, -2, 0] : (!stencil.temp<66x68x60xf64>) -> f64
    %6 = addf %4, %5 : f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])
  stencil.store %3 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  return
  }
```

So everything is fine, but I felt it is overrestricted and its own kind of confusing;
For example, to make sure it has bounds, `store`'s field has to be defined by a `cast`.
Also, why restrict to dynamically sized input buffers to the restrict all computations to known bounds, and risk to lose related optimizations or correctness?

Similarly, in the original dialect, `stencil.store`'s temp had to be defined by an `apply` or `combine`. This is more to enforce properties on the lowering (it makes sure the `store` is just pointing a value-semantics computation to a buffer, to retrieve when lowering so it directly lowers to side-effects on that buffer.)

Now, if we want to imagine more value-semantics combinations, say:
```mlir
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.load %0([-1, -2, 0] : [65, 66, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<66x68x60xf64>
  %3 = stencil.load %1([-1, -2, 0] : [65, 66, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<66x68x60xf64>

%out:2 = for %time = %0 to %T step %1 iter_args(%arg2 = %2, %arg3 = %3) -> (!stencil.field<70x70x60xf64>, !stencil.field<70x70x60xf64>) {

  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<66x68x60xf64>) -> !stencil.temp<64x64x60xf64> {
    %4 = stencil.access %arg2 [-1, 2, 0] : (!stencil.temp<66x68x60xf64>) -> f64
    %5 = stencil.access %arg2 [1, -2, 0] : (!stencil.temp<66x68x60xf64>) -> f64
    %6 = addf %4, %5 : f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([0, 0, 0] : [64, 64, 60])

  yield %arg3, %arg2
}

  stencil.store %out#0 to %0([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %out#1 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
```

We have to handle specifically scf.for to reason abound bounds forward by the iteration of `stencil.apply`, rather than generically forwarding types carrying them.

Encoding bounds in types also allows to simply express the expected bounds in the function signature, ensuring correctness directly and making sure we never lose the constant bounds related optimizations, not needing `stencil.cast` anymore:

```mlir

func @simple(%arg0: !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>, %arg1: !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) attributes {stencil.program} {
  %2 = stencil.load %arg0 : (!stencil.field<[-3,67]x[-3,67]x[0,60]xf64>) -> !stencil.temp<[-1,65]x[-2,66]x[0,60]xf64>
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<[-1,65]x[-2,66]x[0,60]xf64>) -> !stencil.temp<[0,64]x[0,64]x[0,60]xf64> {
    %4 = stencil.access %arg2 [-1, 2, 0] : (!stencil.temp<[-1,65]x[-2,66]x[0,60]xf64>) -> f64
    %5 = stencil.access %arg2 [1, -2, 0] : (!stencil.temp<[-1,65]x[-2,66]x[0,60]xf64>) -> f64
    %6 = addf %4, %5 : f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  }
  stencil.store %3 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<[0,64]x[0,64]x[0,60]xf64> to !stencil.field<[-3,67]x[-3,67]x[0,60]xf64>
  return
  }
```

Finally, note the new signature of `stencil.store`, which now encodes the bounds of both its `stencil.temp` and `stencil.field`. Before this, the verifier itself had to make sure it could access a known stencil operation defining them both, to read bounds from it and check they are sound.\
Now, we can for example shape infer first - still needing to reason in the scope of the stencil dialect at this point, sure - but *after* inference, we can insert arbitrary operations in the def-use chain: `stencil.store` keeps its types and is happy to check bounds locally.

# In-depth points

## `stencil.cast` (dynamic input size) is potentially hurtful

Here's an example (in xDSL, the point being, it is dangerous with current MLIR pipelines):

```mlir
func.func @stencil_init_float(%0 : f64, %1 : !stencil.field<?x?x?xf64>) {
    %2 = stencil.cast %1 : !stencil.field<?x?x?xf64> -> !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    %3 = stencil.apply(%4 = %0 : f64) -> (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>) {
      %5 = arith.constant 1.000000e+00 : f64
      %6 = arith.addf %4, %5 : f64
      stencil.return %6 : f64
    }
    stencil.store %3 to %2 ([1, 2, 3] : [65, 66, 63]) : !stencil.temp<[1,65]x[2,66]x[3,63]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    func.return
}
```

A straight forward first pipeline `xdsl-opt stencil-cast.mlir -p convert-stencil-to-ll-mlir | mlir-opt -p "builtin.module(fold-memref-alias-ops)"` folds away the cast and makes this:

```mlir
#map = affine_map<()[s0] -> (s0 + 3)>
module {
  func.func @stencil_init_float(%arg0: f64, %arg1: memref<?x?x?xf64>) {
    %cst = arith.constant 1.000000e+00 : f64
    %c63 = arith.constant 63 : index
    %c66 = arith.constant 66 : index
    %c65 = arith.constant 65 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%arg2, %arg3, %arg4) = (%c1, %c2, %c3) to (%c65, %c66, %c63) step (%c1, %c1, %c1) {
      %0 = arith.addf %arg0, %cst : f64
      %1 = affine.apply #map()[%arg2]
      %2 = affine.apply #map()[%arg3]
      %3 = affine.apply #map()[%arg4]
      // Dynamic memref store!
      memref.store %0, %arg1[%1, %2, %3] : memref<?x?x?xf64>
      scf.yield
    }
    return
  }
}
```

The loop bounds are hardcoded(as they are in OEC), but the func is accepting any memref now, and will just segfault on smaller buffers!
Also, even if passed the correct size or bigger, it now uses dynamic memref accesses, preventing both constant folding pointer arithmetic and lowering ABIs to simple pointers.
Anyway, the whole semantics is to operate on those fixed bounds (otherwise, let's go parametric and constent fold later?).

With bounds in types, we can just write:

```mlir
func.func @stencil_init_float(%0 : f64, %1 : !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>) {
    %3 = stencil.apply(%4 = %0 : f64) -> (!stencil.temp<[1,65]x[2,66]x[3,63]xf64>) {
      %5 = arith.constant 1.000000e+00 : f64
      %6 = arith.addf %4, %5 : f64
      stencil.return %6 : f64
    }
    stencil.store %3 to %1 ([1, 2, 3] : [65, 66, 63]) : !stencil.temp<[1,65]x[2,66]x[3,63]xf64> to !stencil.field<[-3,67]x[-3,67]x[-3,67]xf64>
    func.return
}
```

Which through the same pipeline yields:

```mlir
func.func @stencil_init_float(%arg0: f64, %arg1: memref<70x70x70xf64>) {
    %cst = arith.constant 1.000000e+00 : f64
    %c63 = arith.constant 63 : index
    %c66 = arith.constant 66 : index
    %c65 = arith.constant 65 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%arg2, %arg3, %arg4) = (%c1, %c2, %c3) to (%c65, %c66, %c63) step (%c1, %c1, %c1) {
      %0 = arith.addf %arg0, %cst : f64
      %1 = affine.apply #map()[%arg2]
      %2 = affine.apply #map()[%arg3]
      %3 = affine.apply #map()[%arg4]
      memref.store %0, %arg1[%1, %2, %3] : memref<70x70x70xf64>
      scf.yield
    }
    return
}
```

NB: Sure, we could also just make `stencil.cast` mandatorily take correct static input size too!

## `stencil.combine` in OEC:

Heres a minmal OEC `stencil.combine` example I could write:
```mlir
module  {
  func @combine(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
    %1 = stencil.apply -> !stencil.temp<34x62x60xf64> {
      %cst = constant 1.000000e+00 : f64
      %4 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
      stencil.return %4 : !stencil.result<f64>
    } to ([-2, 2, 0] : [32, 64, 60])
    %2 = stencil.apply -> !stencil.temp<35x62x60xf64> {
      %cst = constant 0.000000e+00 : f64
      %4 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
      stencil.return %4 : !stencil.result<f64>
    } to ([32, 2, 0] : [67, 64, 60])
    %3 = stencil.combine 0 at 32 lower = (%1 : !stencil.temp<34x62x60xf64>) upper = (%2 : !stencil.temp<35x62x60xf64>) ([-2, 2, 0] : [67, 64, 60]) : !stencil.temp<69x62x60xf64>
    stencil.store %3 to %0([-2, 2, 0] : [67, 64, 60]) : !stencil.temp<69x62x60xf64> to !stencil.field<70x70x60xf64>
    return
  }
}
```

Here's the output of OEC's shape inference:

```mlir
module  {
  func @combine(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
    %1 = stencil.apply -> !stencil.temp<34x62x60xf64> {
      %cst = constant 1.000000e+00 : f64
      %4 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
      stencil.return %4 : !stencil.result<f64>
    } to ([-2, 2, 0] : [32, 64, 60])
    %2 = stencil.apply -> !stencil.temp<35x62x60xf64> {
      %cst = constant 0.000000e+00 : f64
      %4 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
      stencil.return %4 : !stencil.result<f64>
    } to ([32, 2, 0] : [67, 64, 60])
    %3 = stencil.combine 0 at 32 lower = (%1 : !stencil.temp<34x62x60xf64>) upper = (%2 : !stencil.temp<35x62x60xf64>) ([-2, 2, 0] : [67, 64, 60]) : !stencil.temp<69x62x60xf64>
    stencil.store %3 to %0([-2, 2, 0] : [67, 64, 60]) : !stencil.temp<69x62x60xf64> to !stencil.field<70x70x60xf64>
    return
  }
}
```
The bounds do not have to start at 0 in any output or input.

It infers both inputs to have exactly the output's size on other dims, and just enough on the indicated dim (0/first here) from the constant-time known index (32 here) to the upper or lower bound for upper or lower operands respectively.

i.e., on the indicated dim, with the indicated index as I, and output bounds on this dim as [lb,ub], the bounds on this dim of `lower` is [lb,I] and the bounds on this dim of `upper` is [I,ub].