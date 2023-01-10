import itertools
import subprocess
import time
from typing import Any, Callable, TypeVar, cast
from memory_profiler import memory_usage

# Just used for printing
glb_opsize: int = 0
glb_nesting_ratio: float = 0.0
glb_pass_name = ""
glb_statistics = ""
csv_name = "bench_results.csv"

F = TypeVar('F', bound=Callable[..., Any])


def bench(function: F) -> F:

    def wrapper(*args, **kwargs):
        t1 = time.time()
        memory = memory_usage((function, args, kwargs),
                              include_children=True,
                              max_usage=True)
        # function(*args, **kwargs)
        t2 = time.time()
        print(
            f"Config: num_ops:{glb_opsize}, nesting:{glb_nesting_ratio}, pass: {glb_pass_name}, time:{(t2 - t1)} seconds, memory: {memory} MB, statistics: {glb_statistics}"
        )
        with open(csv_name, 'a') as fd:
            fd.write(
                f'{glb_opsize};{glb_nesting_ratio};{glb_pass_name};{(t2 - t1)};{memory};{glb_statistics}\n'
            )

    return cast(F, wrapper)


def get_program(name: str, op_count: int, nesting_ratio: float) -> bytes:
    program = subprocess.run(
        f'python xdsl/tools/generate-big-program -e {name} -c {op_count} -n {nesting_ratio}'
        .split(),
        capture_output=True)
    generated_program = program.stdout

    return generated_program


def analyze_locality(program: bytes):
    locality = subprocess.run(
        f'python xdsl/tools/analyze_rewriting_locality.py'.split(),
        input=program,
        capture_output=True)
    return locality.stdout.decode('utf-8').strip()


def benched_execution(configs: list[tuple[int, float, str]],
                      repetitions: int = 1,
                      keep_module_refs: bool = True):

    @bench
    def execute_rewriting(program: bytes, pass_: str) -> bytes:
        keep_module_refs_str = " --keep-copied-module" if keep_module_refs else ""
        rewritten = subprocess.run(
            f'python xdsl/tools/xdsl-opt-constant-fold-slow.py -p {pass_}{keep_module_refs_str}'
            .split(),
            input=program,
            capture_output=True)


# f"{max(rewriting_localities)}; {statistics.mean(rewriting_localities)}; {statistics.median(rewriting_localities)}; {statistics.stdev(rewriting_localities)}"

    with open(csv_name, 'w') as fd:
        fd.write(
            'opsize;nesting;pass;time;memory;localityMax;localityMean;localityMedian;localityStdev;len\n'
        )
    for op_size, nesting_ratio, pass_ in configs:
        for _ in range(repetitions):
            global glb_opsize, glb_nesting_ratio, glb_pass_name, glb_statistics
            glb_opsize = op_size
            glb_nesting_ratio = nesting_ratio
            glb_pass_name = pass_

            generated_program = get_program('bool_nest', op_size,
                                            nesting_ratio)
            glb_statistics = analyze_locality(generated_program)
            mean_locality = float(glb_statistics.split(';')[1])
            # print(
            #     f"ratio: {mean_locality/op_size} size:{op_size}, nesting:{nesting_ratio},  statistics: {glb_statistics}, pass: {pass_},"
            # )
            execute_rewriting(generated_program, pass_)


def main():
    all_configs = list(
        itertools.product(
            [3000],
            # [10, 30, 100, 300, 1000, 3000],  #, 30000, 100000, 300000],
            # [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
            # [
            #     0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
            #     0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
            # ],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [
                "bool-nest-clone", "bool-nest-composable",
                "bool-nest-no-backtracking"
            ]))
    # example_config = list(
    #     itertools.product(
    #         [30, 100, 300],  #, 30000, 100000, 300000],
    #         [0.2],
    #         ["bool-nest-composable"]))

    # rewriting locality should be near: 13
    # mirror_oec_config = list((100, 0.15), (300, 0.21), (1000), (3000))

    benched_execution(all_configs, repetitions=15)


if __name__ == "__main__":
    main()
