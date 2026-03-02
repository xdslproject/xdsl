# Generating executables through MLIR

xDSL can interoperate with MLIR as its backend. As this
requires an installation, and therefore a compilation, of a lot of the LLVM
project and MLIR, this functonality is not distributed with xDSL by default. To
actually leverage from this functionality, first clone and build MLIR. Please
follow: <https://mlir.llvm.org/getting_started/>

Next, `mlir-opt`, `mlir-translate` and `clang` need to be in the path:

```bash
export PATH=<insert-your-path>/llvm-project/build/bin:$PATH
```

Given an input file `input.mlir`, that contains IR with only the mirrored dialects
found in `xdsl/dialects` (arith, builtin, cf, func, llvm, memref, and scf), run:

```bash
mlir-opt --convert-scf-to-cf --convert-cf-to-llvm --convert-func-to-llvm \
         --convert-arith-to-llvm --expand-strided-metadata --normalize-memrefs \
         --memref-expand --fold-memref-alias-ops --finalize-memref-to-llvm \
         --reconcile-unrealized-casts input.mlir | mlir-translate --mlir-to-llvmir > tmp.ll
```

The generated `tmp.ll` file contains LLVM IR, so it can be directly passed to
the clang compiler. Notice that a `main` function is required for clang to
build. The functionality is tested with the MLIR distributed with LLVM 21.1.1.
