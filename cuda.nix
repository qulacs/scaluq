{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  packages = with pkgs; [
    clang-tools
    cmake
    gcc
    gdb
    kokkos
    ninja
    python312
    python312Packages.numpy
    python312Packages.scipy
    python312Packages.nanobind
    cudaPackages.cudatoolkit
  ];
  shellHook = ''
    export INCLUDEPY=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')
    export NANOBIND=${pkgs.python312Packages.nanobind}
    export PYTHON=${pkgs.python312}
  '';
}
