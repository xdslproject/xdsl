from abc import ABC


class Trait(ABC):
    """
    Defines properties and traits of Operations.
    """


class NoSideEffect(Trait):
    """
    Defines that an Operation does not have side effects.
    """
