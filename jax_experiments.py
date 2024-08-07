import marimo

__generated_with = "0.7.11"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    import jax
    import jax.numpy as jnp
    return jax, jnp


@app.cell
def __():
    return


@app.cell
def __():

    from jax import random
    return random,


@app.cell
def __(get_linalg_module_str, jax, jnp, random):
    def matmul(A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray):
        return A @ B

    matmul_params = (2, 4, 3)
    matmul_shapes = ((matmul_params[0], matmul_params[2]), (matmul_params[2], matmul_params[1]), (matmul_params[0], matmul_params[1]))

    key = jax.random.key(42)

    matmul_data = tuple(random.uniform(key, shape) for shape in matmul_shapes)

    matmul_jit = jax.jit(matmul, donate_argnames="C", keep_unused=True)

    get_linalg_module_str(matmul_jit, matmul_data)
    return (
        key,
        matmul,
        matmul_data,
        matmul_jit,
        matmul_params,
        matmul_shapes,
    )


@app.cell
def __(matmul, matmul_data):
    matmul(*matmul_data), matmul(*matmul_data)
    return


@app.cell
def __():
    from jaxlib.mlir.dialects import mhlo
    from jaxlib.mlir.passmanager import PassManager
    from jax._src.interpreters import mlir

    def get_linalg_module_str(func, args):
        lowered = func.lower(*args)

        mhlo_module = lowered.compiler_ir(dialect="mhlo")
        
        # print(mhlo_module)
        
        with lowered.compiler_ir(dialect="stablehlo").context as ctx:
            ctx.append_dialect_registry(mlir.upstream_dialects)
            # ctx.load_all_available_dialects()
            # mhlo.register_mhlo_dialect(ctx)
            mhlo.register_mhlo_passes()
            pipeline = PassManager.parse(f"builtin.module(func.func(hlo-legalize-to-linalg))")
            pipeline.run(mhlo_module.operation)
        
        mhlo_module_str = f"{mhlo_module}"

        return mhlo_module_str
    return PassManager, get_linalg_module_str, mhlo, mlir


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
