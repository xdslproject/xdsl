import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    from xdsl.utils import marimo as xmo

    return (xmo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 2: Emitting Basic IR

    Now that we're familiar with our language and the AST, let's see how xDSL can
    help to compile Toy.

    ## Introduction: Multi-Level Intermediate Representation

    xDSL leverages the MLIR representation of a program. MLIR specifies a text format of
    this representation, which is useful for debugging and interoperation. For example,
    all the text in this format you'll see in this tutorial can be executed with the
    Toy language as compiled in the (MLIR Toy Tutorial)[https://mlir.llvm.org/docs/Tutorials/Toy/].

    Let's take a quick look at the textual IR representation of our example program:
    """)
    return


@app.cell
def _():
    from toy.compiler import parse_toy

    from xdsl.printer import Printer

    example = """
    def main() {
      var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
      var b<3, 2> = [1, 2, 3, 4, 5, 6];
      var c<2, 3> = b;
      var d = a + c;
      print(d);
    }
    """

    toy_0 = parse_toy(example)
    Printer().print_op(toy_0)
    return (Printer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you might have noticed, some parts look very similar to the Toy program above, and some
    things are added in. Let's look at the structure of an operation in the MLIR output before
    taking a close look at exactly what's going on.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## IR Syntax In Detail

    MLIR is designed to be a completely extensible infrastructure; there is no
    closed set of attributes (think: constant metadata), operations, or types. MLIR
    supports this extensibility with the concept of Dialects. Dialects provide a grouping
    mechanism for abstraction under a unique `namespace`.

    In MLIR, `Operations` are the core unit of abstraction and computation, similar in many
    ways to LLVM instructions. Operations can have application-specific semantics and can be
    used to represent all of the core IR structures in LLVM: instructions, globals (like
    functions), modules, etc.

    Here is the MLIR assembly for the Toy `transpose` operations:

    ```mlir
    # magic command not supported in marimo; please file an issue to add support
    # %t_tensor = "toy.transpose"(%tensor): (tensor<2x3xf64>) -> tensor<3x2xf64>
    ```

    Let's break down the anatomy of this MLIR operation:

    -   `%t_tensor`

        *   The name given to the result defined by this operation (which includes
            [a prefixed sigil to avoid collisions](https://mlir.llvm.org/docs/LangRef/#identifiers-and-keywords)).
            An operation may define zero or more results (in the context of Toy, we
            will limit ourselves to single-result operations), which are SSA values.
            The name is used during parsing but is not persistent (e.g., it is not
            tracked in the in-memory representation of the SSA value).

    -   `"toy.transpose"`

        *   The name of the operation. It is expected to be a unique string, with
            the namespace of the dialect prefixed before the "`.`". This can be read
            as the `transpose` operation in the `toy` dialect.

    -   `(%tensor)`

        *   A list of zero or more input operands (or arguments), which are SSA
            values defined by other operations or referring to block arguments.

    -   `(tensor<2x3xf64>) -> tensor<3x2xf64>`

        *   This refers to the type of the operation in a functional form, spelling
            the types of the arguments in parentheses and the type of the return
            values afterward.

    Shown here is the general form of an operation. As described above,
    the set of operations in MLIR is extensible. Operations are modeled
    using a small set of concepts, enabling operations to be reasoned
    about and manipulated generically. These concepts are:

    -   A name for the operation.
    -   A list of SSA operand values.
    -   A list of attributes.
    -   A list of types for result values.
    -   A source location for debugging purposes.
    -   A list of successors blocks (for branches, mostly).
    -   A list of regions (for structural operations like functions).

    ## Defining Toy Operations

    Now that we have a `Toy` dialect, we can start defining the operations. This
    will allow for providing semantic information that the rest of the system can
    hook into. As an example, let's walk through the creation of a `toy.constant`
    operation. This operation will represent a constant value in the Toy language.

    ```mlir
     %4 = "toy.constant"() {value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    ```

    This operation takes zero operands, a dense elements attribute named `value`
    to represent the constant value, and returns a single result of RankedTensorType.
    Let's take a look at the full definition and step through it in detail.
    """)
    return


@app.cell
def _(Printer):
    from typing import TypeAlias

    # The builtin dialect is a collection of Operations, and Attributes that are expected
    # to be useful for most compilers, such as floating-point numbers, integers,
    # arrays, tensors, and more.
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, Float64Type, TensorType, f64

    # The xdsl.ir module implements the things we've mentioned in this chapter,
    # especially the equivalents of MLIR concepts.
    # xdsl.irdl provides a declarative Python API for Operation definitions
    from xdsl.irdl import IRDLOperation, attr_def, irdl_op_definition, result_def
    from xdsl.utils.exceptions import VerifyException

    TensorTypeF64: TypeAlias = TensorType[Float64Type]


    # A decorator to help implement some methods required by xDSL
    @irdl_op_definition
    class ConstantOp(IRDLOperation):
        """
        Constant operation turns a literal into an SSA value. The data is attached
        to the operation as an attribute. For example:

        ```mlir
          %0 = "toy.constant"() {"value" = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
        ```
        """

        # Every operation has a name. The format is `dialect_name`.`operation_name`
        name = "toy.constant"

        # Attributes are defined using OpAttr, and can specify a type constraint on the attribute
        value = attr_def(DenseIntOrFPElementsAttr[Float64Type])

        # The result type annotation uses `Annotated`, a type in the `typing` module that
        # allows for runtime annotation of types with arbitrary values. xDSL leverages
        # this annotation to populate a `verify()` method that will signal if there is
        # a type mismatch during construction.
        res = result_def(TensorTypeF64)

        def __init__(self, value: DenseIntOrFPElementsAttr):
            super().__init__(result_types=[value.type], attributes={"value": value})

        # Operations can provide helper constructors for ease of use
        @staticmethod
        def from_list(data: list[float], shape: list[int]) -> "ConstantOp":
            value = DenseIntOrFPElementsAttr.from_list(TensorType(f64, shape), data)
            return ConstantOp(value)

        def verify_(self) -> None:
            if not self.res.type == self.value.type:
                raise VerifyException(
                    "Expected value and result types to be equal: "
                    f"{self.res.type}, {self.value.type}"
                )


    Printer().print_op(ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Constructor helper

    ```
        @staticmethod
        def from_list(data: list[float], shape: list[int]):
            value = DenseIntOrFPElementsAttr.tensor_from_list(data, f64, shape)

            return ConstantOp.create(result_types=[value.type],
                                     attributes={"value": value})
    ```

    Operations tend to have helper methods for constructing them, that call into the generic
    constructors on the Operation class. In this case, the client passes in a flat python
    `list` of `float`s for the data, and a shape definition. These get converted to the
    attribute and result type that the `create` method expects as input.

    ### Custom verifier

    ``` python
        def verify_(self) -> None:
            resultType = self.res.type
            value = self.value
            if not isinstance(resultType, TensorType):
                raise VerifyException("Expected result type to be `TensorTypeF64`")

            if not isinstance(value, DenseIntOrFPElementsAttr):
                raise VerifyException(
                    "Expected value type to be instance of `DenseIntOrFPElementsAttr`"
                )

            if resultType.get_shape() != value.shape:
                raise VerifyException(
                    "Expected value and result to have the same shape")
    ```

    One thing to notice here is that all of our Toy operations are printed using the
    generic assembly format. This format is the one shown when breaking down
    `toy.transpose` at the beginning of this chapter. MLIR allows for operations to
    define their own custom assembly format, either or imperatively via C++. Defining a custom
    assembly format allows for tailoring the generated IR into something a bit more readable
    by removing a lot of the fluff that is required by the generic format. Let's walk through
    an example of an operation format that we would like to simplify.

    This capability will soon be added to xDSL also, and will be interoperable with the MLIR
    format definitions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Another Look at the Generated Toy IR

    Let's construct the same Toy program using the Python API:
    """)
    return


@app.cell
def _(Printer):
    from toy.dialects import toy

    from xdsl.builder import Builder
    from xdsl.dialects.builtin import FunctionType, ModuleOp


    @ModuleOp
    @Builder.implicit_region
    def module_op():
        main_type = FunctionType.from_lists([], [])

        @Builder.implicit_region
        def main() -> None:
            a_0 = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
            a = toy.ReshapeOp(a_0, [2, 3]).res
            b_0 = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [6]).res
            b = toy.ReshapeOp(b_0, [3, 2]).res
            c = toy.ReshapeOp(b, [2, 3]).res
            d = toy.AddOp(a, c).res
            toy.PrintOp(d)
            toy.ReturnOp()

        toy.FuncOp("main", main_type, main)


    Printer().print_op(module_op)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First thing we see is that the whole program is wrapped in `builtin.module`. Modules
    roughly represent a parsed input file. Nested one level deep is a `toy.func`, representing
    the `main` function defined in the source. Inside it are a list of instructions.

    The next four lines of MLIR ops correspond to the first two lines of the Toy program.

    ``` MLIR
    # magic command not supported in marimo; please file an issue to add support
    # %0 = "toy.constant"() {"value" = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    # magic command not supported in marimo; please file an issue to add support
    # %1 = "toy.reshape"(%0) : (tensor<2x3xf64>) -> tensor<2x3xf64>
    # magic command not supported in marimo; please file an issue to add support
    # %2 = "toy.constant"() {"value" = dense<[1, 2, 3, 4, 5, 6]> : tensor<6xf64>} : () -> tensor<6xf64>
    # magic command not supported in marimo; please file an issue to add support
    # %3 = "toy.reshape"(%2) : (tensor<6xf64>) -> tensor<3x2xf64>
    ```

    ``` Python
    var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
    var b<3, 2> = [1, 2, 3, 4, 5, 6];
    ```

    Because the types on the left of the `=` operator might not correspond to the type of the
    literal, the reshape operations are inserted. Most of the time the shapes will match,
    and the reshape will be redundant. Let's take a look at how to optimise our code to remove
    redundant reshapes using the xDSL infrastructure.
    """)
    return


if __name__ == "__main__":
    app.run()
