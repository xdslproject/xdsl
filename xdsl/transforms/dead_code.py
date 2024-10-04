from xdsl.ir import Operation, SSAValue
from xdsl.traits import IsTerminator, MemoryEffectKind, SymbolOpInterface, get_effects


def is_trivially_dead(op: Operation):
    """
    Returns if the operation has no observable effect.
    """
    return (
        all(not result.uses for result in op.results)
        and (not op.get_trait(IsTerminator))
        and (not op.get_trait(SymbolOpInterface))
        and result_only_effects(op)
    )


def result_only_effects(rootOp: Operation) -> bool:
    """
    Returns if we can ensure the operation would have no observable effect beyond its
    returned values.

    cf MLIR's WouldOpBeTriviallyDead:
    https://mlir.llvm.org/doxygen/namespacemlir.html#a655db45ed8c23d04d5ed5ee0abe041ad

    We have one key difference here:
    - MLIR discard any allocation from an operation on its own result for this analysis
    - xDSL discard any allocation effect of any nested operation on any value defined
    by the root operation or its children.
    """
    effects = get_effects(rootOp)
    # If the operation has unknown effect, we safely assume it has observable ones
    return effects is not None and all(
        # Read-only effect will not affect other operations
        e.kind == MemoryEffectKind.READ
        # Allocation of values defined by this operation or its children will not
        # affect other operations
        or (
            e.kind == MemoryEffectKind.ALLOC
            and isinstance(v := e.value, SSAValue)
            and rootOp.is_ancestor(v.owner)
        )
        for e in effects
    )
