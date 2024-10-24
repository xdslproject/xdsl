# Stencil rationale

## Current state and differences to the Open Earth Compiler dialect

The current dialect already has slight differences to the one implemented in the Open Earth Compiler. This is currently documented in [rationale/stencil-bounds.md]().

## Next steps

We want the stencil dialect to work more on value-semantics. A few documents explain this, starting by [rationale/value-semantics.md](), which links to the others.

## Road map

- ### A bufferized stencil dialect
  To relax existing constraints, trivially blocking any advanced value-semantics usage, I'm planning to split the stencil lowering in two phases:
  - #### Bufferization:
    Going from the curent state (buffers -> load -> values -> compute -> values -> store -> buffers) to a bufferized state, i.e., computation on buffers directly (buffers -> compute -> compute)

  - #### Conversion:
    Going from a bufferized state to existing dialects (scf, memref, ...) as of now, but greatly simplified by separating bufferization.

  This already yields multiple benefits: This trivially handles in-place computations, and both reading and writing to a buffer in general.

  Documented in [rationale/value-lowering.md]().

- ### Extend stencil's bufferization to loops
  As a way to finally iterate completely in value-semantics, extend stencil's bufferization process to loops. Doing so just reusing existing MLIR bufferization is most likely achievable, but I tried a few things, and it is complex without switching the whole dialect to Destination Passing Style. I'm open to discussion on this, but assumed a smaller footprint on the dialect is desirable and thought about just adding loops to a stencil-specific bufferization implementation for now.

  Documented in [rationale/value-lowering.md]().

- ### Add a `stencil.extend` operation
  It would be similar intuitionally to `stencil.combine`, with a bit more semantics, useful to reason about splitting a value-semantics stencil iteration for distributed memory parallelism.

  Documented in [rationale/stencil-combine.md](), motivated in [rationale/distribution.md]().

- ### Value semantics dmp and distribution
  Currently dmp and automatic distribution rely on the mixed semantics. Small tweaks would be needed to make them work on value-semantics, and would naturally yield similar benefits.