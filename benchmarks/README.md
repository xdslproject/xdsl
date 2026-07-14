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

1. Add a new `time_*` benchmark function or method.

   Benchmarks can be defined as methods on a class, or as standalone functions
   at module scope. When several benchmarks share a `setup` method on the same
   class, inherit `BenchmarkClass` from `bench_utils`. If a timed call may
   consume or alter state prepared in `setup`, leave it on the `BenchmarkClass`
   default. If each call is independent of the last — for example it only reads
   immutable inputs or allocates fresh objects — decorate it with
   `@safe_to_repeat`.

   For example, to benchmark loading a new dialect you might write either:

   ```python
   import importlib

   import xdsl.dialects.newdialect
   from benchmarks.bench_utils import BenchmarkClass, safe_to_repeat

   class ImportDialects(BenchmarkClass):
       @safe_to_repeat
       def time_newdialect_load(self) -> None:
           """Time loading the `newdialect` dialect."""
           importlib.reload(xdsl.dialects.newdialect)
   ```

   or as a standalone function at module scope:

   ```python
   import importlib

   import xdsl.dialects.newdialect
   from benchmarks.bench_utils import safe_to_repeat

   @safe_to_repeat
   def time_newdialect_load() -> None:
       """Time loading the `newdialect` dialect."""
       importlib.reload(xdsl.dialects.newdialect)
   ```

   Standalone functions do not inherit `BenchmarkClass`; see the docstrings on
   `BenchmarkClass` and `safe_to_repeat` in `bench_utils` for how repeated
   timing is configured. Functions and methods named `time_*` are also picked up
   automatically by [airspeed velocity](https://asv.readthedocs.io/en/latest/)
   for tracking performance over time.

2. If the test operates on an xDSL operation or other workload which needs to be set up,
this is best added in `benchmarks/workloads.py` as a new class method on `WorkloadBuilder`.

3. For local profiling, associate the benchmark with a name in the `profile` call at
   the bottom of the file.

   For example, for a module-level benchmark function:

   ```python
   from bench_utils import BenchmarkFunction, profile

   profile(
       {
           # [snip...]
           "Dialects.newdialect_load": BenchmarkFunction(time_newdialect_load),
           # [snip...]
       }
   )
   ```
