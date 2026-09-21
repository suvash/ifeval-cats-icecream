{
  description = "Python + pip dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312; # change version here if you want a different one
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.nodejs
            pkgs.uv
            python
            python.pkgs.pip
            python.pkgs.virtualenv
          ];

          shellHook = ''
            # Create a local venv on first run so pip installs stay project-local
            if [ ! -d .venv ]; then
              ${python}/bin/python -m venv .venv
            fi
            source .venv/bin/activate

            # Ensure the 'prime' package is installed
            if ! python -c "import prime" 2>/dev/null; then
              pip install prime
            fi

            echo "Python $(python --version) ready. pip $(pip --version)"
          '';
        };
      });
}
