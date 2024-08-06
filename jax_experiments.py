import marimo

__generated_with = "0.7.11"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


app._unparsable_cell(
    r"""
    import jax
    import jax.numpy as jnp
    from jaxlib.mlir.dialects import mhlo
    from jaxlib.mlir.passmanager import PassManager
    from jax._src.interpreters import mlir

    @jax.jit
    def hello(A: jnp.ndarray, B: jnp.ndarray):
        return A @ B

    a_shape = (2, 3)
    b_shape = (3, 4)

    from jax import random

    seed = 1701
    key = jax.random.key(seed)

    a_data = random.uniform(key, a_shape)
    b_data = random.uniform(key, b_shape)

    lowered = hello.lower(a_data, b_data)

    mhlo_module = lowered.compiler_ir(dialect=\"mhlo\")

    print(mhlo_module)

    with lowered.compiler_ir(dialect=\"stablehlo\").context as ctx:
        ctx.append_dialect_registry(mlir.upstream_dialects)
        # ctx.load_all_available_dialects()
        # mhlo.register_mhlo_dialect(ctx)
        mhlo.register_mhlo_passes()
        pipeline = PassManager.parse(f\"builtin.module(func.func(hlo-legalize-to-linalg))\")
        pipeline.run(mhlo_module.operation)

    mhlo_module_str = f\"{mhlo_module}\"
    mhlo_module_str s
    """,
    name="__",
)


@app.cell
def __():
    return


@app.cell
def __(a_data, b_data, lowered):
    lowered.compile()(a_data, b_data)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
