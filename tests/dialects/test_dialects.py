from xdsl.dialects import get_all_dialects
from xdsl.traits import MemoryEffect, NoMemoryEffect


def test_op_class_names():
    """
    Make sure that all operation class names match our convention of having an "Op"
    suffix.
    """
    all_dialects = get_all_dialects()
    malformed_op_names = tuple(
        (op.name, op.__name__)
        for dialect_factory in all_dialects.values()
        for op in dialect_factory().operations
        if op.__name__[-2:] != "Op"
    )

    assert not malformed_op_names


def test_no_memory_effects_exclusive():
    """
    Make sure that ops that declare NoMemoryEffects don't also have other traits that
    may have memory effects.
    """
    all_dialects = get_all_dialects()

    for dialect_factory in all_dialects.values():
        for op in dialect_factory().operations:
            effects_traits = tuple(
                trait for trait in op.traits.traits if isinstance(trait, MemoryEffect)
            )
            if any(isinstance(t, NoMemoryEffect) for t in (effects_traits)):
                if other_traits := tuple(
                    trait
                    for trait in effects_traits
                    if not isinstance(trait, NoMemoryEffect)
                ):
                    assert not other_traits, op.name
