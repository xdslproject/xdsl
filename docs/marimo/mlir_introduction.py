import marimo

__generated_with = "0.14.11"
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
            * Specify the types of their operands and results (i32 for all operations here)
    
    
        To try to understand this more, you can change the following example, and see how the generated MLIR code change.
        Try to assign constants to variables, and keep one operator per line (this will be explained later). Also, try to use booleans (`true`, `false`, `&&`, `||`) and comparisons (`==`, `!=`, `<`, `>`, `<=`, `>=`).
    
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
        ## Single-Static Assignment (SSA)
    
        Show this code:
        ```
        c = (3 + 4) + 5
        ```
        * Make people add more operators to it.
        * Show the form we get (SSA form)
    
        Explain that we cannot reassign variables.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Applying compilation passes
    
        Now that we have code, we can apply optimizations on it!
    
        Show the following optimizations:
        * DCE
        * CSE
        * Constant folding
    
        Explain
        * Explain each what they do, and how they change the code.
        * Let people try them out on their own code.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Control flow with Regions
    
        * scf.if example
            * Explain with a figure how this is executed
            * Let people try it with nested ifs, etc...
            * Explain yield
            * Let them try simple optimizations
    
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
    mo.md(r"")
    return


if __name__ == "__main__":
    app.run()
