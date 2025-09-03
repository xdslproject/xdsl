import marimo

__generated_with = "0.15.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Introduction to MLIR

    This notebook introduces the basics of interacting with MLIR through a simple DSL to manipulate arrays.
    The notebook will explain the core concepts around MLIR, show how to read and understand MLIR code, and how to use existing transformations to build a compilation pipeline.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## A first example

    We will start by defining a small language to manipulate lists of integers. This language will be very simple for now, and will be extended throughout the notebook to introduce more concepts. The language is compatible with Rust syntax, and here is a first example:

    ```rust
    let x = 2;
    let y = 3;
    x + y
    ```

    All variables are immutable, and we only support integer literals, addition and multiplication for now.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    We have written a small compiler that translates this simple language to MLIR IR. Here is the MLIR code generated for the example above:

    ```mlir
    %x = arith.constant 2 : i32
    %y = arith.constant 3 : i32
    %x_plus_y = arith.addi %x, %y : i32
    ```

    Note the following:

    * An MLIR program is a sequence of operations (arith.constant, arith.addi in this example)
    * Variables are prefixed with a `%` sign
    * Operations:
        * Have a name (arith.constant for constants, and arith.addi for addition)
        * Produce results (all produce one result here, but they can produce zero or multiple)
        * Have a list of operands (the add has two operands %x and %y, and the constants have none)
        * Can have additional static information (2 and 3 in the arith.constant)
        * Specify some of the types of their operands and results (i32 for all operations here)


    To try to understand this more, you can change the following example, and see how the generated MLIR code change.
    Try to use booleans (`true`, `false`, `&&`, `||`) and comparisons (`==`, `!=`, `<`, `>`, `<=`, `>=`).

    ```rust
    let a = true;
    let b = false;
    a && b;
    ```

    Some things you might have noticed:

    * Booleans are represented as `i1` (a single bit integer)
    * All comparisons operations are encoded as `arith.cmpi` operations, with an opcode (eq, ult, ule, ...) and two operands
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Static Single-Assignment (SSA)

    MLIR IR uses single static-assignment form (SSA), meaning that each variable are defined once. Since our language does not have mutability, the effect is that we cannot "shadow" variables.

    Look how the following code is implemented in MLIR IR:
    ```rust
    let c = 4 + 5;
    let c = c + 2;
    c
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Applying compilation passes

    Now that we have showed some MLIR IR code, let's see how we can manipulate MLIR IR with passes. Here are a few important MLIR IR passes that are relevant for most compilers:

    * `dce` (Dead code elimination): This pass removes code that is not used.
    * `cse` (Common subexpression elimination): This pass merges operations that are exactly the same.
    * `constant-fold-interp` (Constant folding): This pass constant fold operations, so `3 + 4` gets rewritten to `7`.

    Here is a code snippet where you can apply a pass pipeline to MLIR code, and see the effect of each pass. Try to understand what each pass does, and look how reordering passes may have an effect on the resulting operations.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Control flow with Regions

    Now that we have showed how computations work, let's look at how to model control-flow.

    ### `scf.if`

    As a simple first example of control-flow, let's look at a ternary condition. The following example computes the minimum between `x` and `y`:

    ```rust
    let x = 5
    let y = 7
    if x < y then {x} else {y}
    ```

    We can see the following in the generated MLIR IR:

    * The `if` is represented using an `scf.if` operation. `scf` stands for "structured control-flow".
    * `scf.if` contains two **regions**. A region is a section of code that contains another list of operation (we will explain this more in details later).
    * Only one region is executed, this logic comes from `scf.if`. In general, operations that have regions decide when they are executed through compilation passes.
    * Each region ends with an `scf.yield`, this is the "value" that is leaving the region when the region gets executed, and that gets returned by the `scf.if` operation.

    Try to change this program, and look at the effect of different optimization passes!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Adding lists to our DSL

    ### Adding abstractions with dialects

    Now that we have seen most of the core concepts of MLIR IR, let's now add support for lists in our DSL.
    Our list DSL has the following operations:

    * Creating a list from a range (`x..y`)
    * Getting the length of a list (`list.len()`)
    * Mapping a function over a list (`list.map(|x| x + 1)`)

    In order to represent these lists and operations in MLIR IR, we will create our custom new operations and types.
    This is done through defining what's called a **dialect**, a namespace for a set of operations and types.

    Here is an example of a program using lists, feel free to modify it and see the generated MLIR code:
    ```mlir
    let a = 0..10;
    let c = a.map(|x| x + a.len());
    c
    ```

    Note the following:

    * The list type is represented as `!list.list`. This is how custom types are represented in MLIR IR `!dialect.type`.
    * The custom operations, as well as the custom type all start with `list`, which is the name of our dialect.
    * `list.map` uses a region to represent the function to be applied on each element of the list. This region has an argument,
        `x`, which is the element of the list being processed. The `list.yield` operation is used to return the new value for
        the element.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Custom optimizations for our dialect

    One of the main advantages of defining a custom dialect for our lists, is that we can now define simple optimizations that are specific to our dialect, and that would be hard to do otherwise.
    For instance, we can extend the `canonicalize` pass to understand how to optimize `list.map` operations.

    ```mlir
    let a = 0..10;
    let b = a.map(|x| x + 0);
    let c = a.len();
    c
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### `scf.for`

    * scf.for example
        * Show how this is compiled to scf.for
        * Same things as above, this should be clear why regions are useful
        * Show LICM example

    * Show one example of a function
        * Could be cool to explain how a function can be defined

    ## Control flow with Block (and lowering to LLVM)

    * Compilation to basic blocks
        * Mention phi nodes in one sentence
        * Show how this is compiled to basic blocks
        * Give basic idea of dominance, but not too much
        * Show why regions are useful for optimizations, especially with larger examples
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Our own list dialect

    Add lists in the language now

    * Loop fusion for instance
    * length of map?

    Introducing the list dialect:
    * Show the operations, let people play with it and lower it.
    * Show how we can constant fold at a high level
    * Show some optimizations
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r""" """)
    return


if __name__ == "__main__":
    app.run()
