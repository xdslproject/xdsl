#! /bin/bash

# Get object file for Open Earth Compiler's laplace implementation.
mlir-opt --lower-affine --arith-expand --convert-scf-to-cf --expand-strided-metadata --convert-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts ./benchmarks/dialects/experimental/stencil/laplace_oec.mlir | mlir-translate --mlir-to-llvmir | llc --filetype=obj -o laplace_oec_lowered.o

# Lower xDSL's laplace implementation so that we can deal with memrefs inside the final benchmark file.
xdsl-opt ./benchmarks/dialects/experimental/stencil/laplace_xdsl.mlir -t mlir -p stencil-shape-inference,convert-stencil-to-ll-mlir | mlir-opt -canonicalize > laplace_xdsl_ll.mlir

# Get object file for xDSL's laplace implementation.
mlir-opt --arith-expand --convert-scf-to-cf --convert-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts laplace_xdsl_ll.mlir | mlir-translate --mlir-to-llvmir | llc --filetype=obj -o laplace_xdsl_lowered.o

# Get object file used for benchmarking and comparing both implementations.
mlir-opt --lower-affine --arith-expand --convert-scf-to-cf --convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts ./benchmarks/dialects/experimental/stencil/laplace_xdsl_oec.mlir | mlir-translate --mlir-to-llvmir | llc --filetype=obj -o laplace_xdsl_oec_lowered.o

# Link above object files and libmlir_c_runner_utils to get the resultant executable.
gcc -O3 -o laplace_benchmark laplace_xdsl_oec_lowered.o laplace_oec_lowered.o laplace_xdsl_lowered.o -lmlir_c_runner_utils

# Run resultant executable to generate benchmark.
./laplace_benchmark

# Delete all generated object files.
rm -rf *.o

# Delete intermediate laplace_xdsl file.
rm -rf laplace_xdsl_ll.mlir

# Delete final executable.
rm -rf laplace_benchmark
