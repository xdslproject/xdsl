import marimo

__generated_with = "0.11.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from xdsl.irdl import (
        irdl_op_definition,
        irdl_attr_definition,
        ParametrizedAttribute,
        IRDLOperation,
        attr_def,
        result_def,
        operand_def,
        region_def,
        traits_def,
    )
    return (
        IRDLOperation,
        ParametrizedAttribute,
        attr_def,
        irdl_attr_definition,
        irdl_op_definition,
        mo,
        operand_def,
        region_def,
        result_def,
        traits_def,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # The Multi-Level Approach to Compilation

        Most programming languages are rather far removed from assembly code. As a result, translating a language directly into Assembly is both difficult (easy to make mistakes) and inefficient (generated Assembly code is slow!). Therefore, most compilers go through many intermediate representations / languages (IRs), which represent different layers of abstractions, to provide an efficient translation down to Assembly code - which is also feasible for compiler engineers to write.

        Although an effective approach, over the years it has become apparent that many compilers are repeatedly defining similar intermediate languages. Given the separate architecture of different compilers, it has been quite difficult to reuse code between architectures.

        Previous projects such as LLVM have provided a common abstraction layer for different compilers to reuse - however this is only one layer, and compilers of different domains are still having to repeat significant amounts of work to reach the LLVM level.

        LLVM is also not necessarily the ideal layer to perform optimisations on - thus many compilers will also implement their own implementations on custom IRs before they reach LLVM.

        xDSL is a framework that makes it easy to:

        1. Define custom abstraction layers for compilers
        2. Reuse abstractions from other compilers
        3. Encode domain-specific knowledge (i.e. optimisations) within these abstraction layers

        xDSL takes inspiration from MLIR, which is a framework based upon similar ideas, and is implemented in C++.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # General Concepts

        The kind of IRs that xDSL builds are SSA-based - a restriction placed on IRs which makes it easy for analyzing the code.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Static Single Assignment (SSA) 
        The kind of programming languages xDSL defines have the $\textit{static single assignment}$ (SSA) property. It is a property on variables that means they can be assigned a value only once, and once assigned a value, they cannot be modified. This is a common restriction used within compilers, as it enables many compiler optimisations to be performed.

        In an SSA-based language, values are generated by the language's constructs, and old values may be referred to in the construction of new values.

        <!-- give an example of SSA code snippet -->

        Further reading: https://en.wikipedia.org/wiki/Static_single-assignment_form
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# xDSL""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What is an IR, anyway?

        xDSL is a framework for modelling/specifying IRs. Before we go into the details of xDSL, it is perhaps useful to take a step back and consider what exactly an IR is!

        Code can be represented in a human-readable textual format - however strings are difficult for compilers to handle. An IR is an *equivalent*, structured representation of code convenient for machines to parse through. Importantly, it is solely a representation of the code - is not directly executable on a processor.

        Languages are often designed for particular tasks and domains, which allows certain optimisations or types of reasoning to take place more easily within the language. A compiler might want to optimise the code in many different ways, and therefore can utilise multiple languages (represented as IR within the compiler) during its translation into machine code.

        Wikipedia has a nice article as a starting point for further reading: https://en.wikipedia.org/wiki/Intermediate_representation
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Programming Language Syntax
        All programming languages will have a set of "language constructs", or, the _vocabulary_ available within the programming language.

        These constructs can be composed together, forming "expressions" - which are sentences that _could_ be evaluated to a value. 

        Just like in English, though, vocabulary (words) cannot be composed together arbitrarily - they should follow the syntactical rules of the language. For programming languages, this is known as the _syntax_ of the language, and is often specified in terms of a [grammar](https://en.wikipedia.org/wiki/Formal_grammar).

        Here are some examples of building blocks within various languages:  

        - Arithmetic: a language that describes arithmetic might have constructs, including:
            - `+, -, x, /, round, max`  
            - The values of this language are real numbers (floats, say).
            - One syntax rule in this language is that each construct should take in either values, or other valid expressions.  
        - Python: includes too many constructs to fit in this code block, but some examples include:  
            - `for, if, while, assert, def, class, import`  
            - Since Python is object-oriented, values in Python language are objects.  
            - A valid expression in Python would follow the [Python grammar](https://docs.python.org/3/reference/grammar.html), a rather large document specifying all syntax rules of Python.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Programming Language Semantics
        In English, a sentence being grammatically correct is not enough for it to have _meaning_. The meaning of sentences is embedded in the intuition of English speakers, and speakers need to generally agree on the meaning (semantics) of sentences in order to communicate.

        Similarly, for programming languages, the grammar is not enough to convey meaning. There needs to be some semantics of the language which is generally agreed across the users of the programming language and the compiler implementations.

        Programmers make use of the language semantics to write programs which do what they think it does, and compilers utilize the semantics to optimize the programs whilst preserving their semantics - after all, the transformation should not change the meaning of what the programmer wrote!

        For programming languages, the semantics could be assigned to a mathematical function, unambiguously defining the meaning for all sentences in the programming languages.

        However, in industrial programming languages, semantics is often an intuitive notion that is embedded in the minds of language users and within the compiler's transformations (which is preserved).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Programming Language

        With the above two definitions, a programming language's definition simply consists of two parts:  

        1. the syntax
        2. the semantics
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## xDSL: A Tool for Designing and Implementing Languages for Compilers

        Following from the previous definition of a programming language, the corresponding parts of a compiler IR consists of:  

        - The representation of the IR (i.e. the syntax - what constructs there are, how do they compose, what is the data structure to use in the implementation?)
        - The transformations performed on the IR (i.e. the implied semantics: utilizing the expressive powers of the IR in its domain for optimizations) 

        xDSL provides the tools for modelling IRs, where:  

        - Modelling the IR's syntax is done via defining the _operations_, and _attributes_ within the dialect
        - Modelling the IR's transformation is done via specifying _rewrite patterns_ on the IR

        To be able to specify the syntax and the transformations of any IR, it needs to be general enough to fit the wide range of potential designs and uses of language constructs.

        Below, we will introduce the concepts used within xDSL.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Operations: Modelling General Language Constructs

        Operations are units used within xDSL to model a "language construct", i.e. a vocabulary available within the language. 

        A set of operations together would then describe the syntax of the language. 

        An operation is the combination of:   <!-- make it possible to click on the links to go to respective blocks perhaps -->

        1. A name: the name of the operation
        2. A (possibly empty) list of _regions_
        3. A (possibly empty) list of _operands_
        4. A (possibly empty) set of _attributes_
        5. A (nonempty) list of results

        Operations would take in some previously defined SSA values in the program (as operands), and return as a result new SSA value(s).

        The above definition of an operation may seem rather terse - this is because an operation is designed to be general enough to describe all kinds of constructs within a programming language. 

        And so we will provide examples of operations below for more intuition. Moreover, please refer to the individual definitions for more information.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Regions

        A _region_ represents a sequence of instructions with potentially non-linear control flow.

        That is, within a region, the program could execute not just from top-to-bottom, but it may go into different branches or loops. A region therefore corresponds to a control flow graph (_CFG_) - where the nodes are blocks, and edges are the flow of control: we have an edge when a block passes control to another block.

        In xDSL, a _region_ consists of:  

        1. A list of _blocks_
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Blocks

        As commonly seen in compiler theory, a block represents a sequence of instructions that must execute from top-to-bottom, one after the other. 

        At the end of the block, the control flow is handed over to one of its potential _successors_ (which one exactly is dependent on the precise values computed, which is done at runtime)

        What's different about blocks in xDSL is that blocks contain operations, which themselves could contain regions, and regions can contain blocks once again. 

        Thus, the structure of the IR is recursive within xDSL.

        In xDSL, a block is the combination of:  

        1. A list of _block arguments_
        2. A list of _operations_
        3. A (possibly empty) list of _successor_ _blocks_
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Block Arguments

        Instead of using [$\phi$ nodes](https://www.cs.princeton.edu/~appel/papers/ssafun.pdf), xDSL uses _block arguments_ to decide the value to use for computation when input values could come from several different branches.

        These are two equivalent ways to solve the same problem, though, as explained in the earlier linked [article](https://www.cs.princeton.edu/~appel/papers/ssafun.pdf) on $\phi$ nodes.

        In xDSL, block arguments is the combination of:  

        1. An index, denoting the argument's position within the block
        2. A type, i.e. the type of the argument
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Operands

        Operands are the arguments to an operation. This allows the operation to use previously calculated values, which is essential for imperative programs.

        Note that the operands are different to block arguments in that:  

        - _operands_ within an operation refers to a unique previously defined SSA values, whereas
        - _block arguments_ are more akin to function arguments, and do not specify a unique previously defined SSA value to use

        In xDSL, operands are defined as:  

        1. An SSA value: that is, the previously defined SSA value used in this operand
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Attributes

        Attributes are data, in a predetermined format, which are associated with operations. The format of attributes for a particular language come from the language specification. 

        The data it stores must be information which is known at compile time.

        There are several uses of attributes with xDSL, some examples include:  

        - **_type information_**: in xDSL, the types of operation's results are stored as attributes
        - storage for code analysis: [data flow analysis](https://clang.llvm.org/docs/DataFlowAnalysisIntro.html) passes computes properties such as liveness and range of variables, which relies on being able to store and update information within the IR
        - operations whose semantics is dependent on attributes: such as 
            - [convolution operations](https://mlir.llvm.org/docs/Dialects/TOSA/#tosaconv2d-mlirtosaconv2dop) (semantics depends on its stride and padding attributes), 
            - [comparison operations](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpi-mlirarithcmpiop) (semantics depends on its "predicate" attribute, which specifies exactly what comparison to make, i.e. signed less than, unsigned greater than or equals, and so on)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Dialects

        Languages in xDSL are known as `dialects` - which is a grouping of related operations and attributes. For instance,  

        - The `scf` (structured control flow) dialect groups together 
            - Operations such as: for/while loops, reduce, yield, if/else statements
            - Attributes such as: no new attributes in this dialect! It instead uses attributes/types from _other_ dialects
        - The `arith` (arithmetic) dialect groups together
            - Operation such as: comparison, integer/floating-point addition/multiplication, or/xor/and logical bitwise operations, and so on
            - Attributes such as: predicates to decide the mode of comparison, flags to determine mode of floating point computation used
        - The `async` (asynchronous) dialect groups together operations commonly seen in async applications, as well as associated attributes (types).
            - Operations such as: asynchronous calls, waits, async functions
            - Attributes such as: coroutine reference types, future types

        Dialects can be _combined_, such that the combined dialect have the operations and attributes from all constituent dialects.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Rewrite Patterns

        Rewrite pattern is a tool to transform IRs defined using xDSL. 

        It makes it easy to specify peephole optimizations. Rewrite patterns specify a _source_ pattern, and a transformation, which applies itself to the IR by:  

        - matching a _source_ type pattern within the IR
        - updating the IR by applying the predefined transformation

        Multiple rewrite patterns can be applied and composed together in compiler passes.

        A function signature is worth a thousand words... :)

        ```py
        class RewritePattern(ABC):
            \"""
            A side-effect free rewrite pattern matching on a DAG.
            \"""

            @abstractmethod
            def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
                \"""
                Match an operation, and optionally perform a rewrite using the rewriter.
                \"""
                # here, determine if the operation matches our pattern
                # and define a transformation to be applied on the IR making use of the 
                # `rewrite` as an API for changing the IR.
                ...
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# xDSL Examples""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. An SQL Dialect

        As a first example, we'll describe an SSA IR for describing SQL queries, with ability to query and filter on tables. This will be a simple IR which covers two things: 
        * reading from tables (queries like `%table = select(table_name)`)
        * filtering results (`%filtered_table = filter(%table, {filter_conditions})`)

        Each `select` and `filter` constructs returns an object of `Bag` type, representing a query's results. 

        The results of the query/filtering operations are stored in SSA variables (whose names are prefixed with a `%`)

        The language allows us to perform SQL queries - for instance, the one below:
        ```SQL
        SELECT * FROM T WHERE T.a > 5 + 5
        ```

        would look something like this in our SSA IR for SQL:
        ```
        %table = select("T")
        %filtered = filter(%table, (lambda t: t.a > 5 + 5))
        ```

        A compiler for our language should be able to do some optimizations on the code as well. For instance, the above SSA form could be transformed like so:
        ```
        %table = select("T")
        %filtered = filter(%table, (lambda t: t.a > 10))
        ```
        where the constants `5+5` have been folded.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### xDSL Modelling

        Now let's consider how we would model the IR with xDSL.

        The two features of the IR, `select` and `filter`, can be modelled as _operations_: they take in _operands_, and return an SSA value - fitting operations very well.

        Let's start the modelling in xDSL with what we've decided so far:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    example_code_text1 = """@irdl_op_definition
    class SelectOp(IRDLOperation):
        name = "sql.select"
        ...

    @irdl_op_definition
    class FilterOp(IRDLOperation):
        name = "sql.filter"
        ...
    """

    mo.md(f"{mo.ui.code_editor(example_code_text1, language="python", disabled=True)}")
    return (example_code_text1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In xDSL, the dialect definitions are made easy through a framework called IRDL - which reduces much of the boilerplate when defining operations.

        The operation tooling is implemented as Python decorators, as seen in the `@irdl_op_definition` annotation.

        As mentioned above, the dialect introduces a new type, called `Bag`, which represents the return value from queries.

        In xDSL, types can be implemented as _attributes_ like so:
        """
    )
    return


@app.cell
def _(ParametrizedAttribute, irdl_attr_definition):
    @irdl_attr_definition
    class Bag(ParametrizedAttribute):
        name = "sql.bag"
    return (Bag,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Once again, here we make use of IRDL to help us define attributes, as seen in the `@irdl_attr_definition` annotation.

        With our types, defined, we can now specify the _operands_ for our operations:
        """
    )
    return


@app.cell
def _(
    Bag,
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    result_def,
):
    from xdsl.dialects.builtin import StringAttr  # (1)


    @irdl_op_definition
    class SelectOp(IRDLOperation):
        name = "sql.select"
        table_name = attr_def(StringAttr)  # (1)
        result_bag = result_def(Bag)  # (3)

    def _():
        @irdl_op_definition
        class FilterOp(IRDLOperation):
            name = "sql.filter"
            input_bag = operand_def(Bag)  # (2)
            result_bag = result_def(Bag)  # (3)
            ...

        return

    _()
    return SelectOp, StringAttr


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here comments `#` denotes the new lines we added to the _operation_ definitions.

        Let's go through each of these additions:  

        1. To specify the names of our table (which is a string), we needed a string type. Taking advantage of the xDSL ecosystem, we can reuse other dialects which already have similar features defined.
            - Indeed, the `builtin` dialect already has a `StringAttr` attribute defined, and so we use it here for the name of our table.
            - Notice we have made the choice to model the table name in the `select(table_name)` operation as an _attribute_ (as seen by `OpAttr[StringAttr]` annotation). This means that table names in our IR must be known at compile time. This is just a design choice we've made, and it's possible to have the `table_name` be an _operand_ instead, which is an SSA value, and thus doesn't need to be known at compile time.
        2. We specify the _operands_ of the `filter` operation. Here we are saying that `sql.filter` takes in an SSA value.
            - We are additionally saying that the SSA value needs to be of type `Bag`, which enables type-check capabilities automatically for our IRs, a feature provided by xDSL.
        3. We specify the result of our operations - both of our operations should return another SSA value.
            - We are also putting type constraints on our return types, this allows xDSL to type-check our IRs.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We're not quite done yet though. Or `sql.filter` operation doesn't yet model the function used for the filter:

        ```
        filter(%table, (lambda t: t.a > 5 + 5))
        ```

        The filter can be modelled as a _region_, which, recalling from earlier, is a list of `blocks`. Since a region corresponds to a control-flow graph, it is suitable for representing functions such as the one above.

        So, our `sql.filter` operation now looks like this:
        """
    )
    return


@app.cell
def _(
    Bag,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    region_def,
    result_def,
    traits_def,
):
    from xdsl.traits import NoTerminator


    @irdl_op_definition
    class FilterOp(IRDLOperation):
        name = "sql.filter"
        input_bag = operand_def(Bag)
        filter = region_def()
        result_bag = result_def(Bag)
        traits = traits_def(NoTerminator())
    return FilterOp, NoTerminator


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Sample Code in our SQL Language""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's see how the original SSA code:

        ```
        %table = select("T")
        %filtered = filter(%table, (lambda t: t.a > 5 + 5))
        ```

        would be represented in our IR:
        """
    )
    return


@app.cell
def _(Bag, FilterOp, SelectOp):
    from xdsl.printer import Printer  # xDSL tool to visualize our IR
    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block, Region

    # getting parts of other dialects defined elsewhere
    from xdsl.dialects import builtin, arith

    printer = Printer()

    # initializes a "SelectOp" operation:
    # select("T")
    table = SelectOp.build(
        attributes={"table_name": builtin.StringAttr("T")}, result_types=[Bag()]
    )
    table.verify()  # performs type checking and structural checking (e.g. based on operation definition). This will throw if types are invalid.
    printer.print_op(table)

    # defines a filter region                                     # (1)

    block = Block(arg_types=(builtin.i32,))  # filter argument
    with ImplicitBuilder(block) as (arg0,):
        # arg0 is filter argument
        # all operations created in an ImplicitBuilder context will be added to its block
        const1 = arith.ConstantOp.from_int_and_width(5, 32)
        const2 = arith.ConstantOp.from_int_and_width(5, 32)
        summed = arith.AddiOp(const1, const2)
        # sgt stands for `signed greater than`. xDSL represents internally as a constant attribute "4"
        cmp_result = arith.CmpiOp(arg0, summed, "sgt")
    filter_region = Region(block)

    # initializes a "FilterOp" operation using previous selection result and defined region:
    # filter(%table, (lambda t: t > 5 + 5))
    filtered = FilterOp.build(
        result_types=[Bag()], operands=[table], regions=[filter_region]
    )
    filtered.verify()
    printer.print_op(filtered)
    return (
        Block,
        ImplicitBuilder,
        Printer,
        Region,
        arg0,
        arith,
        block,
        builtin,
        cmp_result,
        const1,
        const2,
        filter_region,
        filtered,
        printer,
        summed,
        table,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We get something like this printed:
        ```
        %0 : !sql.bag = sql.select() ["table_name" = "T"]
        %1 : !sql.bag = sql.filter(%0 : !sql.bag) {
        ^0(%2 : !i32):
          %3 : !i32 = arith.constant() ["value" = 5 : !i32]
          %4 : !i32 = arith.constant() ["value" = 5 : !i32]
          %5 : !i32 = arith.addi(%3 : !i32, %4 : !i32)
          %6 : !i1 = arith.cmpi(%2 : !i32, %5 : !i32) ["predicate" = 4 : !i64]
        }
        ```

        Note that we've mixed in several other dialects here:
        - the `arith` dialect for its numerical operations (`arith.constant`, `arith.addi`, `arith.cmpi`)
        - the `builtin` dialect for its numerical types (`builtin.i32`)

        xDSL generates a `.verify()` function which validates the structural (right number of operands, results, regions), and typing (specified in operation definition) constraints of the operation. Custom verification functions can also be implemented by overriding the `.verify_()` function within the operation definition.

        The filter region represents the lambda `\a -> a > 5+5` by utilizing a single block:
        - We constructed the block by the builder function `Block.from_callable`
        - The argument of the `lambda` is the block arguments
        - The computation of the `lambda` is represented by the block's contents: a list of operations
        - The final operation within a block is a special one. It must be an operation to either:
            - pass control flow to one of the block's _successors_, or
            - be a terminator operation like `scf.yield`, which is akin to a "return".

        So we now have a representation of the original code as an xDSL IR (which are objects in memory).

        This representation allows us to transform and optimize our IR. We'll see a few examples in the section below.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Transforming the xDSL IR""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        By inspecting the IR, we notice that the results of the addition/comparison operations within the `lambda` are already known at compile-time - since their inputs are all known.

        So, we can write a transformation to "fold" the constants to cut down the runtime work.

        We will implement the constant folding transformation as an xDSL `RewritePattern`, and apply this to our IR using xDSL's _RewriteEngine_.
        """
    )
    return


@app.cell
def _(arith):
    from xdsl.pattern_rewriter import (
        PatternRewriter,
        RewritePattern,
        op_type_rewrite_pattern,
    )


    class ConstantFolding(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: arith.AddiOp, rewriter: PatternRewriter):
            # pattern: if both arguments to the Addi operation are from `Constant` operations
            if isinstance(op.lhs.op, arith.ConstantOp) and isinstance(
                op.rhs.op, arith.ConstantOp
            ):
                # transform: replace the operation by calculating the sum of the constants at compile time
                return rewriter.replace_matched_op(
                    arith.ConstantOp.from_int_and_width(
                        op.lhs.op.value.value.data + op.rhs.op.value.value.data,
                        op.lhs.op.value.type.width.data,
                    )
                )
    return (
        ConstantFolding,
        PatternRewriter,
        RewritePattern,
        op_type_rewrite_pattern,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""applying this to our IR:""")
    return


@app.cell
def _(ConstantFolding, filtered, printer):
    from xdsl.pattern_rewriter import (
        PatternRewriteWalker,
        GreedyRewritePatternApplier,
    )


    walker1 = PatternRewriteWalker(
        GreedyRewritePatternApplier([ConstantFolding()]),
        walk_regions_first=True,
        apply_recursively=True,
        walk_reverse=False,
    )
    walker1.rewrite_region(filtered.filter)
    printer.print_op(filtered)
    return GreedyRewritePatternApplier, PatternRewriteWalker, walker1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""the above introduced some variables which are no longer used (e.g. `%3` and `%4`), therefore can be safely removed from the code.""")
    return


@app.cell
def _(
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    arith,
    filtered,
    op_type_rewrite_pattern,
    printer,
):
    class DeadConstantElim(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: arith.ConstantOp, rewriter: PatternRewriter):
            if len(op.result.uses) == 0:
                rewriter.erase_matched_op()


    walker2 = PatternRewriteWalker(
        GreedyRewritePatternApplier([DeadConstantElim()]),
        walk_regions_first=True,
        apply_recursively=True,
        walk_reverse=False,
    )
    walker2.rewrite_region(filtered.filter)
    printer.print_op(filtered)
    return DeadConstantElim, walker2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And now, the transformed filter operation is equivalent to:
        ```
        filter(%table, (lambda t: t.a > 10))
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Summary

        So, we have now defined a mini-dialect using many of xDSL's features and concepts introduced earlier, including:  

        - Operations: we modelled the SQL dialect using `select` and `filter` operations
        - Attributes: we used attributes to encode compile-time available information, from types (the `Bag` type), to constant expressions (e.g. the "cmpi" attribute within `arith.Cmpi`)
        - Mixing dialects: we utilised features from other dialects (`arith`, `scf`, `builtin`)
        - Regions and blocks: we used the operation's _regions_ to model the lambda within `sql.filter` since regions allow nesting of code.
            - The function arguments are modelled as the _block arguments_ of the first block in the region, and the function's code/computation is represented as the block's list of operations.
            - Not all operations need a region, though. For instance `sql.select` does not have a region.
        - xDSL type-checking: xDSL is able to use the typing information within the dialect definition to validate IRs (through `.verify()` calls)
        - Rewrites: we defined two rewrites to transform the xDSL IR, simplifying and optimising the code (`ConstantFolding` and `DeadConstantElim`)
        """
    )
    return


if __name__ == "__main__":
    app.run()
