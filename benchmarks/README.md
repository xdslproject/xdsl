# Benchmarks for xDSL

This directory contains benchmarks and profiling utilities for the xDSL compiler
framework.

## Benchmarking strategy

For a holistic view of xDSL's performance, we provide three levels of
granularity for benchmarking. At the highest level, end-to-end tests such as
running `xdsl-opt` on MLIR files capture the entire compiler pipeline. Below
this, component tests such as lexing MLIR files only capture individual stages
in the compiler pipeline. Finally, microbenchmarks evaluate properties of the
implementation, and are desiged to align with
[existing tests of MLIR](https://www.youtube.com/watch?v=7qvVMUSxqz4).

### List of benchmarks

- End-to-end
  - [x] Empty program
  - [x] Constant folding (with and without verifier)
  - [x] Loop unrolling
  - [ ] CIRCT workload
  - [ ] ASL workload
  - [ ] ...
- Component
  - [x] Lexer
  - [x] Parser
  - [ ] Pattern rewriter
  - [ ] Printer
  - [ ] Verifier
- Microbenchmarks
  - [x] IR traversal (direct block iteration and walking)
  - [x] Dialect loading
  - [x] Import machinery
  - [ ] Extensibility through interface/trait lookups
  - [ ] Operation creation
  - [ ] ...

## Automated regression benchmarking with ASV

> airspeed velocity (asv) is a tool for benchmarking Python packages over their
> lifetime. Runtime, memory consumption and even custom-computed values may be
> tracked. The results are displayed in an interactive web frontend that
> requires only a basic static webserver to host.
>
> -- [ASV documentation](https://asv.readthedocs.io/en/stable/index.html)

Every day by the cron schedule `0 4 * * *`, a GitHub actions workflow is run
using ASV to benchmark the 15 most recent commits to the xDSL repository, and
commit the results to the `.asv/results/github-action` directory of an
[artefact repository](https://github.com/xdslproject/xdsl-bench). Then, the
interactive web frontend is built from these results and all previously
committed results from previous workflow runs, then finally deployed to GitHub
pages.

This web frontend can be found at <https://xdsl.dev/xdsl-bench/>.

## Profiling

In addition to running under ASV, all benchmarks can be profiled with a variety
of tools using the infrastructure in `bench_utils.py`. This provides a simple
CLI when a benchmark file is directly run which through which the tool and
benchmark can be specified. The help page for this CLI is as follows:

```
uv run python3
 BENCHMARK.py --help
usage: BENCHMARK.py [-h] [-o OUTPUT] [-t TEST] [-s]
                     {timeit,snakeviz,viztracer,flameprof} ...

positional arguments:
  {timeit,snakeviz,viztracer,flameprof}
    timeit              use the timeit
    snakeviz            use the SnakeViz profiler
    viztracer           use the VizTracer profiler
    flameprof           use the flameprof profiler

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        the directory into which to write out the profile
                        files
  -t TEST, --test TEST  the name of the test to run
  -q, --quiet           don't show the profiler's UI
```

### Examples

To use `timeit` to get the average runtime of the lexer on
`apply_pdl_extra_file.mlir`:

```bash
uv run python3 component__lexer.py -t time_lexer__apply_pdl_extra_file timeit
```

To use `viztracer` to profile the end-to-end optimisation of a constant folding
workload

```bash
uv run python3 end_to_end.py -t time_end_to_end_opt__constant_folding viztracer
```

### Extensibility

This infrastructure can be modified to support further profilers by adding
further subcommands and command implementations to the `bench_utils.py` file.
