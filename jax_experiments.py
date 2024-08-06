import marimo

__generated_with = "0.7.11"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __(get_linalg_module_str):
    import jax
    import jax.numpy as jnp

    def hello(A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray):
        return A @ B

    a_shape = (2, 3)
    b_shape = (3, 4)
    c_shape = (2, 4)

    from jax import random

    seed = 1701
    key = jax.random.key(seed)

    a_data = random.uniform(key, a_shape)
    b_data = random.uniform(key, b_shape)
    c_data = random.uniform(key, c_shape)

    hello_jit = jax.jit(hello, donate_argnames="C", keep_unused=True)

    get_linalg_module_str(hello_jit, (a_data, b_data, c_data))
    return (
        a_data,
        a_shape,
        b_data,
        b_shape,
        c_data,
        c_shape,
        hello,
        hello_jit,
        jax,
        jnp,
        key,
        random,
        seed,
    )


@app.cell
def __(a_data, b_data, c_data, hello, hello_jit):
    hello(a_data, b_data, c_data), hello_jit(a_data, b_data, c_data)
    return


@app.cell
def __():
    from jax._src.interpreters import mlir
    from jaxlib.mlir.dialects import mhlo
    from jaxlib.mlir.passmanager import PassManager

    def get_linalg_module_str(func, args):
        lowered = func.lower(*args)

        mhlo_module = lowered.compiler_ir(dialect="mhlo")

        print(mhlo_module)

        with lowered.compiler_ir(dialect="stablehlo").context as ctx:
            ctx.append_dialect_registry(mlir.upstream_dialects)
            # ctx.load_all_available_dialects()
            # mhlo.register_mhlo_dialect(ctx)
            mhlo.register_mhlo_passes()
            pipeline = PassManager.parse(
                "builtin.module(func.func(hlo-legalize-to-linalg))"
            )
            pipeline.run(mhlo_module.operation)

        mhlo_module_str = f"{mhlo_module}"

        return mhlo_module_str

    return PassManager, get_linalg_module_str, mhlo, mlir


@app.cell
def __():
    return


@app.cell
def __(a_data, b_data, lowered):
    lowered.compile()(a_data, b_data)
    return


@app.cell
def __(a_data, b_data, get_linalg_module_str, jax):
    import numpy as np
    from jax import vmap

    def update_buffer_numpy(x, buffer):
        # This function will be called with numpy arrays
        np.add(x, buffer, out=buffer)
        return buffer

    def update_buffer_jax(x, buffer):
        result_shape = jax.ShapeDtypeStruct(buffer.shape, buffer.dtype)
        return jax.pure_callback(update_buffer_numpy, result_shape, x, buffer)

    def update_buffer(x, buffer):
        return x + buffer

    mapped_update = vmap(
        update_buffer,
        in_axes=0,
        out_axes=0,
        # axis_resources={'i': 'mem'}
    )

    get_linalg_module_str(jax.jit(mapped_update), (a_data, b_data))
    return (
        mapped_update,
        np,
        update_buffer,
        update_buffer_jax,
        update_buffer_numpy,
        vmap,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
