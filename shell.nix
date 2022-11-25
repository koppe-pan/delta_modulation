{ pkgs ? import <nixpkgs> {} }:
let
  python =
    (pkgs.python3.withPackages (ps: with ps; with pkgs.python3Packages; [
      jupyter
      ipython

      matplotlib
      numpy
      openpyxl
      pandas
      pytorch
      scikit-learn
      statsmodels
      torchvision
      websockets
      #scikit-rf
      plotly

      # optmization
      pulp

      pysptk
      pyworld
      fastdtw
    ]));
in pkgs.mkShell {
  buildInputs = [
    pkgs.graphviz
    python
  ];
}
