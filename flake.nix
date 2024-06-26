{
  description = "xDSL devshell";

  inputs = {
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
              LD_LIBRARY_PATH = "${stdenv.cc.cc.lib}/lib";
              buildInputs = [
                nodejs_22
                ruff
              ];
            };
          }
    );
}
