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
        in
          {
            devShells.default = with pkgs; mkShell {
              LD_LIBRARY_PATH = lib.makeLibraryPath [ stdenv.cc.cc.lib zlib ];
              buildInputs = [
                uv
                nodejs_22
                llvmPackages_22.llvm
                llvmPackages_22.mlir
                llvmPackages_22.tblgen
              ];
              XDSL_MLIR_OPT = "${llvmPackages_22.mlir}/bin/mlir-opt";
              XDSL_MLIR_TRANSLATE = "${llvmPackages_22.mlir}/bin/mlir-translate";
              XDSL_LLVM_DIFF = "${llvmPackages_22.llvm}/bin/llvm-diff";
            };
          }
    );
}
