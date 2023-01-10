class RegionsOf:
    """
    Used to define regions of the operation together with the `with` block.

    For example, one can represent `scf.if` xDSL/MLIR operation in the front-end
    simly with the following code.

    ```
    x = ...
    with RegionsOf(scf.if(condition)) as (true_region, false_region):
        with true_region:
            x = ...
        with false_region:
            x = ...
    ```
    Here, when lowered to xDSL, `scf.if` yields an udated versions of `x`.

    Another example how one can encode `scf.for`. Again, side-effects are computed
    based on the symbolic values in the enclosed regions.
    ```
    with RegionsOf(scf.for(start, end, step)) as body_region:
        with body_region:
            x = ...
    ```
    """

    def __init__(self, op):
        pass

    def __enter__(self, *args):
        return 

    def __exit__(self, *args):
        pass


def meta(*params):
    """
    Decorator used to mark function arguments as templated arguments. In particular, suitable for template
    metaprogramming (or some kind of partial evaluation). Takes a list of argument names as parameters.

    ```
    @meta("A")
    def foo(A: int):
        return A + 2

    # Original foo is translated into this function.
    def foo_10():
        return 10 + 2
    
    def main():
        foo(10)
    ```
    """
    def decorate(f):
        return f
    return decorate


def block():
    """
    Decorator used to mark function as a basic block.

    ```
    def foo(a: int) -> int:
        bb0(a)

        @block
        def bb0(x: int):
            y: int = x + 2
            bb1(y)
        
        @block
        def bb1(z: int):
            return z
    ```
    """
    def decorate(f):
        return f
    return decorate
