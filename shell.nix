{ nixpkgs ? import <nixpkgs> {}, compiler ? "default" }:

let

  inherit (nixpkgs) pkgs;

  f = { mkDerivation, base, cuda, filepath
      , language-c, stdenv, storable-complex, template-haskell
      , autoconf
      }:
      mkDerivation {
        pname = "cublas";
        version = "0.3.0.0";
        src = ./.;
        libraryHaskellDepends = [
          base cuda filepath language-c storable-complex template-haskell
        ];
	buildDepends = [ autoconf ];
        librarySystemDepends = [ pkgs.cudatoolkit ] ; # [ cublas cusparse ];
        homepage = "https://github.com/bmsherman/cublas";
        description = "FFI bindings to the CUDA CUBLAS and CUSPARSE libraries";
        license = stdenv.lib.licenses.bsd3;
	#shellHook = ''
	#  export CUDA_PATH=${pkgs.cudatoolkit}
	#'';
      };

  haskellPackages = if compiler == "default"
                       then pkgs.haskellPackages
                       else pkgs.haskell.packages.${compiler};

  drv = haskellPackages.callPackage f {};

in

  if pkgs.lib.inNixShell then drv.env else drv
