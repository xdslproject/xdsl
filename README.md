# xDSL

## Prerequisites

To install XDSL you can either clone the Github repository and install the requirements by following:

### Clone and install
```bash
git clone https://github.com/xdslproject/xdsl.git
pip install -e .
# or for the optional requirements
# pip install -e .[extras]
```

### pip installation

```bash
pip install xdsl
```

## Testing

This project includes pytest unit test and llvm-style filecheck tests. They can
be executed using to following commands from within the root directory of the
project:

```bash
# Executes pytests which are located in tests/
pytest

# Executes filecheck tests
lit tests/filecheck
```

## Generating executables through MLIR

xDSL can generate executables using MLIR as the backend. To use this
functionality, make sure to have clang, mlir-opt and mlir-translate installed
and in the PATH. Clang can be installed by standard package managers (apt,
pacman...), for MLIR follow [this](https://mlir.llvm.org/getting_started/) and
run `ninja install` afterward. Given an input file `input.xdsl`, that contains
IR with only the mirrored dialects found in `xdsl/dialects` (arith, builtin,
cf, func, irdl, llvm, memref, and scf), run:

```bash
### Prints MLIR generic from to tmp.mlir
./src/tools/xdsl_opt -t mlir  -o tmp.mlir `input.xdsl`

mlir-opt --convert-scf-to-cf --convert-cf-to-llvm --convert-func-to-llvm --convert-arith-to-llvm --convert-memref-to-llvm --reconcile-unrealized-casts tmp.mlir | mlir-translate --mlir-to-llvmir > tmp.ll
```

The generated `tmp.ll` file contains LLVMIR, so it can be directly passed to a
compiler like clang. Notice that a `main` function is required for clang to
build. Refer to `tests/filecheck/arith_ops.test` for an example. The
functionality is tested with MLIR git commit hash:
74992f4a5bb79e2084abdef406ef2e5aa2024368


## Formatting

All python code used in xDSL uses [yapf](https://github.com/google/yapf) to
format the code in a uniform manner.

To automate the formatting within vim, one can use
https://github.com/vim-autoformat/vim-autoformat and trigger a `:Autoformat` on
save.
