import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 1. Attributes

        Attributes represent compile-time information. They are used for compile-time constants and types, as well as comments/supplementary information not used during the runtime of the program.

        Attributes are immutable, comparable, and hashable.

        In xDSL, attribute classes must have a name that uniquely identifies them.

        There are two kinds of attribute: Data and ParametrizedAttribute.

        ## Data

        `Data` attributes are used to wrap immutable Python objects. Here is an example that defines an attribute that wraps a Python `float`:
        """
    )
    return


@app.cell
def _(Any, AttrParser, Printer, math):
    from xdsl.irdl import irdl_attr_definition
    from xdsl.ir import Data

    # All attribute definitions use this annotation.
    @irdl_attr_definition
    class FloatAttr(Data[float]):
        # The name of the attribute.
        name = "float"

        # All Data attributes must define methods to print and parse to
        # MLIR-compatible syntax.
        @classmethod
        def parse_parameter(cls, parser: AttrParser) -> float:
            with parser.in_angle_brackets():
                return float(parser.parse_number())

        def print_parameter(self, printer: Printer) -> None:
            printer.print_string(f"<{self.data}>")

        # The `__eq__` and `__hash__` methods are usually synthesized, but can
        # be overridden, like in this case.
        def __eq__(self, other: Any):
            # avoid triggering `float('nan') != float('nan')` inequality
            return isinstance(other, FloatAttr) and (
                math.isnan(self.data) and math.isnan(other.data) or self.data == other.data
            )

        def __hash__(self):
            return hash(self.data)

    # The attribute can be instantiated like so:
    FloatAttr(3)
    return Data, FloatAttr, irdl_attr_definition


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## ParametrizedAttribute

        `ParametrizedAttribute` attributes are used to build attributes out of other attributes. Here is an example that represents a floating-point complex type.
        """
    )
    return


@app.cell
def _(FloatAttr, irdl_attr_definition):
    from xdsl.ir import ParametrizedAttribute
    from xdsl.irdl import ParameterDef

    @irdl_attr_definition
    class ComplexFloatAttr(ParametrizedAttribute):
        # ParametrizedAttribute classes must also have a name
        name = "complex"

        # The parameters of this attribute
        real: ParameterDef[FloatAttr]
        imag: ParameterDef[FloatAttr]

        # Custom initializers can be defined
        def __init__(self, real: FloatAttr, imag: FloatAttr):
            # The default initializer argument is a tuple of parameters, in order
            super().__init__((real, imag))

    # The attribute can be instantiated like so:
    ComplexFloatAttr(FloatAttr(3), FloatAttr(4))
    return ComplexFloatAttr, ParameterDef, ParametrizedAttribute


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Exercise: Using Attributes

        As an exercise, implement the following functions, which convert from Python complex numbers and back:
        """
    )
    return


@app.cell
def _(ComplexFloatAttr, FloatAttr):
    def from_python(value: complex) -> ComplexFloatAttr:
        return ComplexFloatAttr(FloatAttr(0), FloatAttr(0))

    def to_python(attr: ComplexFloatAttr) -> complex:
        return -1 + 0j
    return from_python, to_python


@app.cell
def _(from_python):
    # Should print (
    #   `#complex<#float<1>, #float<2>>`,
    #   `#complex<#float<3>, #float<4>>`
    # )
    from_python(1 + 2j), from_python(3 + 4j)
    return


@app.cell
def _(ComplexFloatAttr, FloatAttr, to_python):
    # Should print (
    #   (5+6j),
    #   (7+8j)
    # )
    _a = ComplexFloatAttr(FloatAttr(5), FloatAttr(6))
    _b = ComplexFloatAttr(FloatAttr(7), FloatAttr(8))
    to_python(_a), to_python(_b)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Attribute Types

        The types of runtime values are also represented as attributes that inherit from the `TypeAttribute` class. Only these can be used as types of runtime values, and they are otherwise defined in the same way in xDSL. Here are a couple of definitions of type attributes:
        """
    )
    return


@app.cell
def _(ParameterDef, ParametrizedAttribute, irdl_attr_definition):
    from xdsl.ir import TypeAttribute

    @irdl_attr_definition
    class FloatType(ParametrizedAttribute, TypeAttribute):
        name = "float"

    @irdl_attr_definition
    class ComplexType(ParametrizedAttribute, TypeAttribute):
        name = "complex"
        element_type: ParameterDef[FloatType]

        def __init__(self) -> None:
            super().__init__((FloatType(),))

    FloatType(), ComplexType()
    return ComplexType, FloatType, TypeAttribute


@app.cell
def _(mo):
    mo.md(
        r"""
        As you may have noticed, these print with an `!` before the name, as opposed to the attributes above which began with `#`.
        Note that in MLIR Attributes and Types are distinct, and are in distinct namespaces.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Builtin Types

        xDSL and MLIR provide a number of built-in helpers and components that we encourage to use when possible.
        Conveniently, these are placed in the `builtin` dialect.

        Examples of attributes are:

        * `i32`, a signless integer type of bitwidth 32
        * `4 : i32`, the constant 4 encoded on 32 bits
        * `memref<5xi32>`, a memory reference to 5 32-bits integers

        Please see the documentation for more details.

        **TODO add link to documentation**

        Here is the Python API for the examples above:
        """
    )
    return


@app.cell
def _():
    from xdsl.dialects.builtin import i32, IntegerAttr, MemRefType

    _int = IntegerAttr(4, i32)
    _memref = MemRefType(i32, (5,))

    _int, _memref
    return IntegerAttr, MemRefType, i32


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
