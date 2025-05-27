import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from xdsl.utils import marimo as xmo
    from xdsl.ir import Block, Region, Operation, SSAValue
    from xdsl.builder import Builder, InsertPoint
    from xdsl.dialects import builtin, riscv, riscv_cf, riscv_func
    from xdsl.printer import Printer
    from xdsl.parser import Parser
    from xdsl.context import Context
    from xdsl.backend.riscv.riscv_register_queue import RiscvRegisterQueue
    import difflib
    from xdsl.backend.register_type import RegisterType
    from collections.abc import Iterator
    from collections import defaultdict
    return (
        Block,
        Context,
        Parser,
        RegisterType,
        RiscvRegisterQueue,
        SSAValue,
        builtin,
        defaultdict,
        mo,
        riscv,
        riscv_cf,
        riscv_func,
        xmo,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Register Allocation in RISC-V

    [Register allocation](https://en.wikipedia.org/wiki/Register_allocation) is the process of assigning values to registers.
    This notebook describes the implementation of register allocation in xDSL, with our RISC-V dialect as an example.
    Our implementation is a work in progress, and is currently limited in ways that we'll describe below.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Register Types

    We use `riscv.IntRegisterType` to encode the register that an integer value will be allocated to during the execution of the program.
    In the executable, the index of the register is used, which is encoded as a 5-bit unsigned integer.
    In assembly, the name is used.
    The `RiscvRegisterType` holds both the index and the name of the register:
    """
    )
    return


@app.cell
def _(riscv):
    _x0 = riscv.IntRegisterType.from_name("x0")
    _zero = riscv.IntRegisterType.from_index(0)
    print(_x0, _zero)

    # We use the index to check for equivalence, as the same register can be represented by different names in assembly
    print(_x0.index == _zero.index)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To denote values that we have not yet allocated to a register we set the index to `NoneAttr` and the name to an empty string:""")
    return


@app.cell
def _(riscv):
    _unallocated = riscv.IntRegisterType.unallocated()
    print(f"""{_unallocated}:
      index: {_unallocated.index}
      name: {_unallocated.register_name}
      is allocated: {_unallocated.is_allocated}""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## An Example Function

    Here's a function that is partially allocated, meaning some of the register types have a name in the brackets (`a0`, `a1`), and others don't:
    """
    )
    return


@app.cell(hide_code=True)
def _(Parser, ctx, xmo):
    # The input

    simple_ir = """\
    riscv_func.func @simple(%a : !riscv.reg<a0>, %b : !riscv.reg<a1>) -> !riscv.reg<a0> {
      %c = riscv.add %a, %b : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg
      %d = riscv.add %a, %b : (!riscv.reg<a0>, !riscv.reg<a1>) -> !riscv.reg
      %e = riscv.add %c, %d : (!riscv.reg, !riscv.reg) -> !riscv.reg
      %res_e = riscv.mv %e : (!riscv.reg) -> !riscv.reg<a0>
      riscv_func.return %res_e : !riscv.reg<a0>
    }
    """
    simple_module = Parser(ctx, simple_ir).parse_module()
    simple_module.verify()

    simple_func = next(iter(simple_module.regions[0].blocks[0].ops))
    simple_block = simple_func.body.block

    xmo.module_html(simple_func)
    return simple_block, simple_func, simple_module


@app.cell(hide_code=True)
def _(allocated_register_names, mo, simple_block, simple_func, xmo):
    try:
        _sorted = "{" + ", ".join(sorted(allocated_register_names(simple_block))) + "}"
        _res = str(_sorted)
    except Exception as e:
        _res = f"{type(e).__name__}: {e}"

    _accordion = mo.accordion({"Show input function": xmo.module_html(simple_func)})

    mo.md(
        fr"""
        ### Exercise 1. Allocated Registers

        {_accordion}

        Modify the function below to return the set of register names that are allocated in the block.

        ```
        Expected: {{a0, a1}}
        Result:   {_res}
        ```
        """
    )
    return


@app.cell
def _(Block, SSAValue):
    def iter_operands_and_results(block: Block) -> set[SSAValue]:
        for child in block.ops:
            yield from child.operands
            yield from child.results

    def allocated_register_names(block: Block) -> set[str]:
        return {
            value.type.register_name.data
            for value in iter_operands_and_results(block)
        }

    # Click on 👁️ below to show solution
    return allocated_register_names, iter_operands_and_results


@app.cell(hide_code=True)
def _(Block, iter_operands_and_results, riscv):
    def _allocated_register_names(block: Block) -> set[str]:
        return {
            value.type.register_name.data
            for value in iter_operands_and_results(block)
            if isinstance(value.type, riscv.IntRegisterType) and value.type.is_allocated
        }

    return


@app.cell(hide_code=True)
def _(mo, simple_func, simple_module, unallocated_value_names, xmo):
    try:
        _sorted = "{" + ", ".join(sorted(unallocated_value_names(simple_module))) + "}"
        _res = str(_sorted)
    except Exception as e:
        _res = f"{type(e).__name__}: {e}"

    _accordion = mo.accordion({"Show input function": xmo.module_html(simple_func)})

    mo.md(
        fr"""
        ### Exercise 2. Unallocated Values

        {_accordion}

        Modify the function below to return the set of values that have unallocated register types in the block.

        ```
        Expected: {{c, d, e}}
        Result:   {_res}
        ```
        """
    )
    return


@app.cell
def _(Block, iter_operands_and_results):
    def unallocated_value_names(block: Block) -> set[str]:
        return {
            value.name_hint
            for value in iter_operands_and_results(block)
        }

    # Click on 👁️ below to show solution
    return (unallocated_value_names,)


@app.cell(hide_code=True)
def _(Block, iter_operands_and_results, riscv):
    def _unallocated_value_names(block: Block) -> set[str]:
        return {
            value.name_hint
            for value in iter_operands_and_results(block)
            if isinstance(value.type, riscv.IntRegisterType) and not value.type.is_allocated
        }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Liveness and Conflicts

    One of the key constraints of register allocation is that a register can't hold two values at the same time.
    Another constraint is that we have a limited number of registers, which may be fewer than the number of values.
    This means that we may have to assign multiple values to the same register, but in a way that avoids conflicts where an operation that reads from a register gets a value that wasn't expected.
    Such a conflict would happen if a value was initialised with a certain type before the last use of a value of the same register type.
    In other words, in order to allocate registers correctly, no operation executed between the initialisation of a value and its last use must have a result of the same type.
    """
    )
    return


@app.cell(hide_code=True)
def _(conflicting_value_map, mo, simple_block, simple_func, xmo):
    try:
        _map = conflicting_value_map(simple_block)
        _sorted = {
            k: sorted(_map[k])
            for k in sorted(_map)
        }
        _res = str(_sorted)
    except Exception as e:
        _res = f"{type(e).__name__}: {e}"

    _accordion = mo.accordion({"Show input function": xmo.module_html(simple_func)})

    mo.md(
        fr"""
        ### ~Exercise 3. Conflicting Registers~

        {_accordion}

        Modify the function below to return a map from values to values that it conflicts with. This includes other results of the same operation, and all results created before the value's last use.

        ```
        Expected: {{'c': ['a', 'b'], 'd': ['c'], 'e': [], 'res_e': []}}
        Result:   {_res}
        ```
        """
    )
    return


@app.cell
def _(Block, SSAValue, defaultdict):
    def conflicting_value_map(block: Block) -> dict[str, set[str]]:
        res: defaultdict[SSAValue, set[SSAValue]] = defaultdict(set)
        live_values = set()
    
        for op in reversed(block.ops):
            # When we walk in reverse, the last use of a value is the first use we see!
            live_values.update(op.results)
            for r in op.results:
                res[r].update(live_values)
            live_values.difference_update(op.results)
            live_values.update(op.operands)

        # A value doesn't conflict with itself
        for k in res:
            res[k].remove(k)

        return {
            k.name_hint: {v.name_hint for v in res[k]}
            for k in res
        }
    return (conflicting_value_map,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The objective is to find suitable registers for `%c`, `%d`, and `%e`.
    Note that some of the values already have registers assigned, and we want to keep those assignments.
    The first question is: what are the available registers?

    The RISC-V ISA uses five bits to specify a register, so there are 32 registers in total, these can be referenced by name in assembly by `x0` to `x31`.
    The register encoded by the bit pattern 00000 is the `zero` register, which always holds the value 0, which leaves 31 registers that can be assigned to and read.
    """
    )
    return


@app.cell
def _(riscv):
    x0 = riscv.IntRegisterType.from_name("x0")
    zero = riscv.IntRegisterType.from_index(0)
    print(x0, zero)

    # We use the index to check for equivalence, as the same register can be represented by different names in assembly
    print(x0.index == zero.index)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(RegisterType, mo):
    mo.md(
        rf"""
    ## Register Allocation

    [Register allocation](https://en.wikipedia.org/wiki/Register_allocation) is the process of assigning values to registers.
    This notebook describes the implementation of register allocation in xDSL, with our RISC-V dialect as an example.
    Our implementation is a work in progress, and is currently limited in ways that we'll describe below.

    The pass works on values with a type that inherits from [`{RegisterType.__name__}`](https://xdsl.readthedocs.io/stable/reference/backend/register_type/#xdsl.backend.register_type.RegisterType).
    Values of this type may be _allocated_, meaning that the value will be stored in a specific register at run time (e.g. `a0` or `t1`), or _unallocated_ otherwise.
    The target machine has a limited number of registers, which may be smaller than the number of values in the input IR.
    That means that some values will have to be assigned to the same register.
    In order for register allocation to be correct, no two values must be assigned to the same register at the same time.

    The notion of time here is tricky.
    During the execution of the program, a value with an allocated type being initialised corresponds to a write to a register, and a value being used corresponds to a register read.
    The set of operations that may be executed between a value is initialised and the last time it is used is called the live range.
    """
    )
    return


@app.cell(hide_code=True)
def _(RiscvRegisterQueue, mo, riscv):
    mo.md(
        rf"""
    ## RISC-V Calling Convention

    The RISC-V standard assigns special meaning to the rest of the registers, which limits their use but simplifies interoperation of functions written by hand or compiled separately.
    For example, the `sp` (x{riscv.IntRegisterType.from_name("sp").index.data}) register holds the address of the stack.
    Some registers are specified to hold either function arguments or result values: `a0`, ..., `a7` for integers, `fa0`, ..., `fa7` for floating-point (the contents of the `a` registers not used for return values is not defined at the end of the function call).
    Some registers must hold the same value across function calls (`s0`, ..., `s11`), while others may be overwritten (`t0`, ..., `t6`).

    For convenience, the lists of registers that should and shouldn't be used during register allocation are stored on the `{RiscvRegisterQueue.__name__}` class:
    """
    )
    return


@app.cell
def _(RiscvRegisterQueue):
    print("Available:")
    print(sorted(reg.register_name.data for reg in RiscvRegisterQueue.default_available_registers()))
    print("Reserved:")
    print(sorted(reg.register_name.data for reg in RiscvRegisterQueue.default_reserved_registers()))
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(Context, builtin, riscv, riscv_cf, riscv_func):
    ctx = Context()
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(riscv.RISCV)
    ctx.load_dialect(riscv_cf.RISCV_Cf)
    ctx.load_dialect(riscv_func.RISCV_Func)
    return (ctx,)


if __name__ == "__main__":
    app.run()
