# xDSL

## Prerequisits

To install all required dependencies, execute the following command:

```
pip install -r requirements.txt
```

TODO: check if PYTHONPATH is required or if there exists an easy fix for it.

## Testing

This project includes pytest unit test and llvm-style filecheck tests. They can be executed using to following commands from within the root directory of the project:

```
# Executes pytests which are located in tests/
pytest

# Executes filecheck tests
lit tests/filecheck
```

## Generating executables through MLIR

xDSL can generate executables using MLIR as the backend. To use this functionality, make sure to install the [MLIR Python Bindings](https://mlir.llvm.org/docs/Bindings/Python/). Given an input file `input.xdsl`, that contains IR with only the mirrored dialects found in `src/xdsl/dialects` (arith, memref, func, cf, scf, and builtin), run:

```
### Prints MLIR generic from to tmp.mlir
./src/xdsl/xdsl_opt.py -t mlir  -o tmp.mlir `input.xdsl`

mlir-opt --convert-scf-to-cf --convert-cf-to-llvm --convert-func-to-llvm --convert-arith-to-llvm --convert-memref-to-llvm --reconcile-unrealized-casts tmp.mlir | mlir-translate --mlir-to-llvmir > tmp.ll
```

The generated `tmp.ll` file contains LLVMIR, so it can be directly passed to a compiler like clang.
Notice that a `main` function is required for clang to build. Refer to `tests/filecheck/arith_ops.test` for an example.
The functionality is tested with MLIR git commit hash: 74992f4a5bb79e2084abdef406ef2e5aa2024368


## Formatting

All python code used in xDSL use yapf to format the code in a uniform manner. 

https://github.com/google/yapf

To automate the formatting within vim, one can use https://github.com/vim-autoformat/vim-autoformat and trigger a `:Autoformat` on save.
