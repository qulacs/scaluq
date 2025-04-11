{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            clang-tools
            cmake
            gcc
            gdb
            kokkos
            ninja
            poetry
            python312
          ];
          shellHook = ''
            export INCLUDEPY=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')
          '';
        };
      }
    );
}