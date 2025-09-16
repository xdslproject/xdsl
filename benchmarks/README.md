# Benchmarking infrastructure for xDSL

This directory contains infrastructure for the benchmarking and performance profiling of
the xDSL compiler framework.

The benchmarks are structured such that they can be automatically used by the [airspeed
velocity](https://asv.readthedocs.io/en/latest/), enabling a CI-based web frontend: <https://xdsl.dev/xdsl-bench/>.

In addition to this, helper scripts and infrastructure is provided to run and profile
benchmark performance on a developer's local machine.

## Using benchmarks locally

Benchmarks can be invoked by calling the script that contains them, which has a CLI to
select the benchmarks to run and tools to run them under.
These tools are installed with the `uv` extra group `bench`.
The help menu for this tool can be shown by invoking any script with the `--help` flag:

```text
$ uv run benchmarks/lexer.py --help
usage: lexer.py [-h] [-o OUTPUT] [-q]
                {Lexer.empty_program,Lexer.constant_100,Lexer.constant_1000,Lexer.dense_attr,Lexer.dense_attr_hex,all} [{run,timeit,snakeviz,viztracer,flameprof,pyinstrument,dis}]

positional arguments:
  {Lexer.empty_program,Lexer.constant_100,Lexer.constant_1000,Lexer.dense_attr,Lexer.dense_attr_hex,all}
                        the name of the benchmark to run, `all` to run all benchmarks
  {run,timeit,snakeviz,viztracer,flameprof,pyinstrument,dis}
                        the profiler to use

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   the directory into which to write out the profile files
  -q, --quiet           don't show the profiler's UI
```

We provide some common usage examples below:

### Running a benchmark

```bash
uv run benchmarks/lexer.py Lexer.empty_program run
```

### Running all benchmarks in a file

```bash
uv run benchmarks/lexer.py all run
```

### Timing a benchmark

```bash
uv run benchmarks/lexer.py Lexer.empty_program timeit
```

### Profiling a benchmark

```bash
uv run benchmarks/lexer.py Lexer.empty_program viztracer
```

### Disassembling a benchmark

```bash
uv run benchmarks/lexer.py Lexer.empty_program dis
```

## Adding new benchmarks

1. Create a new method on the relevant benchmark class.

   For example, to add a new dialect to the `ImportDialect` class, you might write:

   ```python
    def time_newdialect_load(self) -> None:
        """Time loading the `newdialect` dialect."""
        importlib.reload(xdsl.dialects.newdialect)
    ```

2. If the test operates on an xDSL operation or other workload which needs to be set up,
this is best added in `benchmarks/workloads.py` as a new class method on `WorkloadBuilder`.

3. Associate the method with a name in the main code at the bottom of the file

   For example, to add this new method, you might write

   ```python
    from bench_utils import Benchmark, profile

    DIALECTS = ImportDialects()
    profile(
        {
            # [snip...]
            "Dialects.affine_load": Benchmark(DIALECTS.time_newdialect_load),
            # [snip...]
        }
    )
    ```
