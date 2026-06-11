{
  description = "xDSL devshell";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };
          llvm = pkgs.llvmPackages_22;
        in
          {
            devShells.default = with pkgs; mkShell {
              LD_LIBRARY_PATH = lib.makeLibraryPath [ stdenv.cc.cc.lib zlib ];
              buildInputs = [
                uv
                nodejs_22
                (symlinkJoin {
                  name = "llvm-mlir-tools";
                  paths = [
                    llvm.mlir
                    llvm.llvm
                    llvm.tblgen
                  ];
                })
              ];
              shellHook = ''
                export XDSL_MLIR_OPT="${llvm.mlir}/bin/mlir-opt"
                export XDSL_MLIR_TRANSLATE="${llvm.mlir}/bin/mlir-translate"
                export XDSL_LLVM_DIFF="${llvm.llvm}/bin/llvm-diff"
              '';
            };
          }
    );
}
