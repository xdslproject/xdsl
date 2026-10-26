# Ground truth for xdsl#6450 step 1

Generated with Homebrew LLVM 23.1.1 (`mlir-opt --version`). CI pins LLVM 22.1.2;
the syntax used here is identical on both.

    mlir-opt matmul_tile.mlir    -transform-interpreter > matmul_tile.out.mlir
    mlir-opt matmul_tile_ij.mlir -transform-interpreter > matmul_tile_ij.out.mlir

- `matmul_tile.mlir`    — tile_sizes [32, 32, 32] -> 3 scf.for loops, 32x32x32 matmul
- `matmul_tile_ij.mlir` — tile_sizes [32, 32, 0]  -> 2 scf.for loops, 32x128 / 128x32 slices
